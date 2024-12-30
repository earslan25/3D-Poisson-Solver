#include "gpu_solver.cuh"
#include "math_solvers.cuh"
#include <iostream>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

// gpu solver and CUDA kernels for the 3D Poisson equation
GPUSolver::GPUSolver(std::shared_ptr<Grid> grid) : 
    grid_(grid) {
    // allocate memory for the solution
    // printf("Initializing GPU data...\n");
    initGpuData();
    cudaCheckError();
}

GPUSolver::~GPUSolver() {
    // free memory
    freeGpuData();
}

__device__
int getNeighbors(unsigned int index, unsigned int NX, unsigned int NY, unsigned int NZ, 
        Neighbors *neighbors, unsigned int *i, unsigned int *j, unsigned int *k, double *phi) {
    // get the i, j, k indices, where index = i + j*NX + k*NX*NY
    *i = index % NX;
    *j = (index / NX) % NY;
    *k = index / (NX*NY);

    // bounds
    if (*i == 0 || *i == NX-1 || *j == 0 || *j == NY-1 || *k == 0 || *k == NZ-1) {
        return -1;
    }
    
    // get the neighbors of the current point
    neighbors->xPos = phi[index + 1];
    neighbors->xNeg = phi[index - 1];
    neighbors->yPos = phi[index + NX];
    neighbors->yNeg = phi[index - NX];
    neighbors->zPos = phi[index + NX*NY];
    neighbors->zNeg = phi[index - NX*NY];

    return 0;
}

__global__
void solveRedBlackKernel(double *phi, double *f, double *phiAnalytic,
        unsigned int NX, unsigned int NY, unsigned int NZ, 
        double dh, double *reducedErrors, unsigned int order) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    tid *= 2;
    tid += order;

    if (tid >= NX*NY*NZ) {
        return;
    }

    // for error reduction
    __shared__ double s_error[1024 / WARP_SIZE];
    double error = 0.0; 

    // get the i, j, k indices and the neighbors
    unsigned int i = 0, j = 0, k = 0;
    Neighbors neighbors;
    int res = getNeighbors(tid, NX, NY, NZ, &neighbors, &i, &j, &k, phi);

    if (res == 0) {
        // solve the equation
        double fCurr = f[tid];
        phi[tid] = (neighbors.xPos + neighbors.xNeg + neighbors.yPos + neighbors.yNeg + 
                neighbors.zPos + neighbors.zNeg - fCurr*dh) / 6.0;

        // compute the error
        error += sqrErr(phi[tid], phiAnalytic[tid]);
    }

    // reduce the error
    __syncwarp();
    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        error += __shfl_down_sync(FULL_MASK, error, offset);
    }

    // write the error to shared memory
    if (threadIdx.x % WARP_SIZE == 0) {
        s_error[threadIdx.x / WARP_SIZE] = error;
    }

    // reduce within the thread block
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int i = 1; i < blockDim.x / WARP_SIZE; i++) {
            error += s_error[i];
        }

        reducedErrors[blockIdx.x] = error;
    }
}

void GPUSolver::solveRB(double &error) {
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    unsigned int totalSize = NX*NY*NZ;
    unsigned int halfSize = totalSize / 2 + totalSize % 2;
    unsigned int arrSize = halfSize * 2;

    double dx = grid_->getDX();
    double dy = grid_->getDY();
    double dz = grid_->getDZ();

    double dh = dx * dy * dz;

    // set up CUDA blocks, 1d, for red-black ordering
    dim3 nthreads(256, 1, 1);
    unsigned int numberOfBlocks = (halfSize + nthreads.x - 1) / nthreads.x;
    dim3 nblocks(numberOfBlocks, 1, 1);

    // error array, set to 0
    double *errors_reduced_h = new double[arrSize];
    memset(errors_reduced_h, 0, arrSize*sizeof(double));

    // allocate memory on GPU for errors
    double *errors_reduced_d;
    cudaMalloc((void**)&errors_reduced_d, arrSize*sizeof(double));

    // solve on GPU
    // launch red kernel
    solveRedBlackKernel<<<nblocks, nthreads>>>(
        phi_d_, f_d_, phiAnalytic_d_,
        NX, NY, NZ,
        dh, errors_reduced_d, 0
    );

    // cudaCheckError();

    // launch black kernel
    solveRedBlackKernel<<<nblocks, nthreads>>>(
        phi_d_, f_d_, phiAnalytic_d_,
        NX, NY, NZ,
        dh, errors_reduced_d + halfSize, 1
    );

    // sync
    cudaDeviceSynchronize();

    // copy errors to host
    cudaMemcpy(errors_reduced_h, errors_reduced_d, 
            arrSize*sizeof(double), cudaMemcpyDeviceToHost);

    // cudaCheckError();

    // reduce the errors
    #pragma omp parallel for reduction(+:error)
    for (unsigned int i = 0; i < arrSize; i++) {
        error += errors_reduced_h[i];
    }

    // free memory
    cudaFree(errors_reduced_d);
    delete[] errors_reduced_h;
}

__global__
void solveJacobiKernel(double *phi, double *f, double *phiAnalytic, double *phiNew,
        unsigned int NX, unsigned int NY, unsigned int NZ, 
        double dh, double *reducedErrors) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= NX*NY*NZ) {
        return;
    }

    // for error reduction
    __shared__ double s_error[1024 / WARP_SIZE];
    double error = 0.0; 

    // get the i, j, k indices and the neighbors
    unsigned int i = 0, j = 0, k = 0;
    Neighbors neighbors;
    int res = getNeighbors(tid, NX, NY, NZ, &neighbors, &i, &j, &k, phi);

    if (res == 0) {
        // solve the equation
        double fCurr = f[tid];
        phiNew[tid] = (neighbors.xPos + neighbors.xNeg + neighbors.yPos + neighbors.yNeg + 
                neighbors.zPos + neighbors.zNeg - fCurr*dh) / 6.0;

        // compute the error
        error += sqrErr(phiNew[tid], phiAnalytic[tid]);
    }

    // reduce the error
    __syncwarp();
    for (unsigned int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        error += __shfl_down_sync(FULL_MASK, error, offset);
    }

    // write the error to shared memory
    if (threadIdx.x % WARP_SIZE == 0) {
        s_error[threadIdx.x / WARP_SIZE] = error;
    }

    // reduce within the thread block
    __syncthreads();
    if (threadIdx.x == 0) {
        for (unsigned int i = 1; i < blockDim.x / WARP_SIZE; i++) {
            error += s_error[i];
        }

        reducedErrors[blockIdx.x] = error;
    }
}

void GPUSolver::swapFaces(double *currPhi) {
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    unsigned int xPosInc = NX - 1;
    unsigned int yPosInc = NX*(NY-1);
    unsigned int zPosInc = NX*NY*(NZ-1);

    double **tempPhi = &currPhi;

    // swap the start pointers of faces to currPhi
    cudaMemcpy(&xNegFace_d_->start, tempPhi, sizeof(double*), cudaMemcpyHostToDevice); 
    cudaMemcpy(&yNegFace_d_->start, tempPhi, sizeof(double*), cudaMemcpyHostToDevice);
    cudaMemcpy(&zNegFace_d_->start, tempPhi, sizeof(double*), cudaMemcpyHostToDevice);
    *tempPhi = currPhi + xPosInc;
    cudaMemcpy(&xPosFace_d_->start, tempPhi, sizeof(double*), cudaMemcpyHostToDevice);
    *tempPhi = currPhi + yPosInc - xPosInc;
    cudaMemcpy(&yPosFace_d_->start, tempPhi, sizeof(double*), cudaMemcpyHostToDevice);
    *tempPhi = currPhi + zPosInc - yPosInc;
    cudaMemcpy(&zPosFace_d_->start, tempPhi, sizeof(double*), cudaMemcpyHostToDevice);
}

void GPUSolver::solveJacobi(double &error) {
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    unsigned int totalSize = grid_->getNX() * grid_->getNY() * grid_->getNZ();

    double dx = grid_->getDX();
    double dy = grid_->getDY();
    double dz = grid_->getDZ();

    double dh = dx * dy * dz;

    // set up CUDA blocks, 1d
    dim3 nthreads(256, 1, 1);
    unsigned int numberOfBlocks = (totalSize + nthreads.x - 1) / nthreads.x;
    dim3 nblocks(numberOfBlocks, 1, 1);

    // error arrays, set to 0
    double *errors = new double[numberOfBlocks];
    memset(errors, 0, numberOfBlocks*sizeof(double));

    // allocate memory on GPU for errors
    double *errors_reduced_d;
    cudaMalloc((void**)&errors_reduced_d, numberOfBlocks * sizeof(double));

    // solve on GPU
    solveJacobiKernel<<<nblocks, nthreads>>>(
        phi_d_, f_d_, phiAnalytic_d_, phiNew_d_,
        NX, NY, NZ,
        dh, errors_reduced_d
    );

    // sync
    cudaDeviceSynchronize();

    // copy errors from GPU to CPU
    cudaMemcpy(errors, errors_reduced_d, numberOfBlocks*sizeof(double), cudaMemcpyDeviceToHost);

    // cudaCheckError();

    // reduce the errors
    #pragma omp parallel for reduction(+:error)
    for (unsigned int i = 0; i < numberOfBlocks; i++) {
        error += errors[i];
    }

    // free memory
    cudaFree(errors_reduced_d);
    delete[] errors;

    // swap phiNew with phi
    std::swap(phi_d_, phiNew_d_);

    // swap face pointers
    swapFaces(phi_d_);
}   

void GPUSolver::copyResultsToCpu(double *phiNew) {
    unsigned int totalSize = grid_->getNX() * grid_->getNY() * grid_->getNZ();
    cudaMemcpy(phiNew, phi_d_, totalSize*sizeof(double), cudaMemcpyDeviceToHost);
}

void GPUSolver::copyResultsToGpu() {
    unsigned int totalSize = grid_->getNX() * grid_->getNY() * grid_->getNZ();
    double *phi = grid_->getPhi();
    cudaMemcpy(phi_d_, phi, totalSize*sizeof(double), cudaMemcpyHostToDevice);
}

void GPUSolver::copyBufferToCpu(double *testBuffer, double *buffer_d, unsigned int size) {
    cudaMemcpy(testBuffer, buffer_d, size*sizeof(double), cudaMemcpyDeviceToHost);
}

double *GPUSolver::getSendBuffer() {
    return sendBuffer_d_;
}

double *GPUSolver::getRecvBuffer() {
    return recvBuffer_d_;
}

__global__
void setSendFacesKernel(Face *face, unsigned int start, double *sendBuffer, unsigned int streamIndex) {
    // 1 thread block per face
    unsigned int i = threadIdx.x;
    unsigned int offset = blockDim.x;

    for (; i < face->dim1; i += offset) {
        unsigned int indexDim1 = i * face->inc1;
        for (unsigned int j = 0; j < face->dim2; j++) {
            // send inner face so adjust start pointer
            double *innerStart = face->start;
            if (streamIndex % 2 == 1) {
                innerStart += face->selfInc;
            } 
            else {
                innerStart -= face->selfInc;
            }
            unsigned int index = indexDim1 + j * face->inc2;
            // write to send buffer incrementing start pointer by 1
            unsigned int bufferIndex = i * face->dim2 + j + start;
            sendBuffer[bufferIndex] = innerStart[index];
        }
    }
}

// copy the face to the send buffer
void GPUSolver::copyFacesToSendBuffer(int (&neighborRanks)[6]) {
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    // Face *allFaces[6] = {xNegFace_d_, xPosFace_d_, yNegFace_d_, yPosFace_d_, zNegFace_d_, zPosFace_d_};
    Face *allFaces[6] = {xPosFace_d_, xNegFace_d_, yPosFace_d_, yNegFace_d_, zPosFace_d_, zNegFace_d_};

    unsigned int start = 0;

    // create streams
    cudaStream_t streams[6];
    for (unsigned int i = 0; i < 6; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 nthreads(256, 1, 1);
    dim3 nblocks(1, 1, 1);
    for (unsigned int i = 0; i < 6; i++) {
        if (neighborRanks[i] != -2) {
            setSendFacesKernel<<<nblocks, nthreads, 0, streams[i]>>>(
                allFaces[i], start, sendBuffer_d_, i);
        }
        start += faceDims_[i];
    }

    // sync
    cudaDeviceSynchronize();
    // cudaCheckError();

    // destroy streams
    for (unsigned int i = 0; i < 6; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

__global__
void setFacesFromRecvKernel(Face *face, unsigned int start, double *recvBuffer) {
    // 1 thread block per face
    unsigned int i = threadIdx.x;
    unsigned int offset = blockDim.x;

    for (; i < face->dim1; i += offset) {
        unsigned int indexDim1 = i * face->inc1;
        for (unsigned int j = 0; j < face->dim2; j++) {
            unsigned int index = indexDim1 + j * face->inc2;
            unsigned int bufferIndex = i * face->dim2 + j + start;
            face->start[index] = recvBuffer[bufferIndex];
        }
    }
}

// copy the recv buffer to the face
void GPUSolver::copyRecvBufferToFaces(int (&neighborRanks)[6]) {
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    // Face *allFaces[6] = {xNegFace_d_, xPosFace_d_, yNegFace_d_, yPosFace_d_, zNegFace_d_, zPosFace_d_};
    Face *allFaces[6] = {xPosFace_d_, xNegFace_d_, yPosFace_d_, yNegFace_d_, zPosFace_d_, zNegFace_d_};

    unsigned int start = 0;

    // create streams
    cudaStream_t streams[6];
    for (unsigned int i = 0; i < 6; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 nthreads(256, 1, 1);
    dim3 nblocks(1, 1, 1);
    for (unsigned int i = 0; i < 6; i++) { 
        if (neighborRanks[i] != -2) {
            setFacesFromRecvKernel<<<nblocks, nthreads, 0, streams[i]>>>(
                allFaces[i], start, recvBuffer_d_);
        }       
        start += faceDims_[i];
    }

    // sync
    cudaDeviceSynchronize();
    // cudaCheckError();

    // destroy streams
    for (unsigned int i = 0; i < 6; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

void GPUSolver::sync() {
    cudaDeviceSynchronize();
}

double *GPUSolver::getPhiGpu() {
    return phi_d_;
}

void GPUSolver::createPlane(double *start, 
        unsigned int dim1, unsigned int dim2, 
        unsigned int inc1, unsigned int inc2, 
        unsigned int selfInc, Face *plane) {
    plane->start = start;
    plane->dim1 = dim1;
    plane->dim2 = dim2;
    plane->inc1 = inc1;
    plane->inc2 = inc2;
    plane->selfInc = selfInc;
}

void GPUSolver::initGpuData() {
    double *phi = grid_->getPhi();
    double *phiNew_d = grid_->getPhiNew();
    double *phiAnalytic = grid_->getAnalyticPhi();
    double *f = grid_->getF();

    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    // create 6 faces, index = i + j*NX + k*NX*NY
    Face xPosFace;
    Face xNegFace;
    Face yPosFace;
    Face yNegFace;
    Face zPosFace;
    Face zNegFace;

    // allocate memory on GPU
    cudaMalloc((void**)&phi_d_, NX*NY*NZ*sizeof(double));
    cudaMalloc((void**)&phiNew_d_, NX*NY*NZ*sizeof(double));
    cudaMalloc((void**)&phiAnalytic_d_, NX*NY*NZ*sizeof(double));
    cudaMalloc((void**)&f_d_, NX*NY*NZ*sizeof(double));

    createPlane(phi_d_ + NX - 1, NZ, NY, NX*NY, NX, 1, &xPosFace);
    faceDims_[0] = xPosFace.dim1 * xPosFace.dim2;
    createPlane(phi_d_, NZ, NY, NX*NY, NX, 1, &xNegFace);
    faceDims_[1] = xNegFace.dim1 * xNegFace.dim2;
    createPlane(phi_d_ + NX*(NY-1), NZ, NX, NX*NY, 1, NX, &yPosFace);
    faceDims_[2] = yPosFace.dim1 * yPosFace.dim2;
    createPlane(phi_d_, NZ, NX, NX*NY, 1, NX, &yNegFace);
    faceDims_[3] = yNegFace.dim1 * yNegFace.dim2;
    createPlane(phi_d_ + NX*NY*(NZ-1), NY, NX, NX, 1, NX*NY, &zPosFace);
    faceDims_[4] = zPosFace.dim1 * zPosFace.dim2;
    createPlane(phi_d_, NY, NX, NX, 1, NX*NY, &zNegFace);
    faceDims_[5] = zNegFace.dim1 * zNegFace.dim2;

    cudaMalloc((void**)&xPosFace_d_, sizeof(Face));
    cudaMalloc((void**)&xNegFace_d_, sizeof(Face));
    cudaMalloc((void**)&yPosFace_d_, sizeof(Face));
    cudaMalloc((void**)&yNegFace_d_, sizeof(Face));
    cudaMalloc((void**)&zPosFace_d_, sizeof(Face));
    cudaMalloc((void**)&zNegFace_d_, sizeof(Face));

    // send and receive buffers for all 6 faces
    cudaMalloc((void**)&sendBuffer_d_, 2*(NX*NY + NX*NZ + NY*NZ)*sizeof(double));
    cudaMalloc((void**)&recvBuffer_d_, 2*(NX*NY + NX*NZ + NY*NZ)*sizeof(double));

    // copy data to GPU
    cudaMemcpy(phi_d_, phi, NX*NY*NZ*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(phiNew_d_, phiNew_d, NX*NY*NZ*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(phiAnalytic_d_, phiAnalytic, NX*NY*NZ*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(f_d_, f, NX*NY*NZ*sizeof(double), cudaMemcpyHostToDevice);

    cudaMemcpy(xPosFace_d_, &xPosFace, sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(xNegFace_d_, &xNegFace, sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(yPosFace_d_, &yPosFace, sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(yNegFace_d_, &yNegFace, sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(zPosFace_d_, &zPosFace, sizeof(Face), cudaMemcpyHostToDevice);
    cudaMemcpy(zNegFace_d_, &zNegFace, sizeof(Face), cudaMemcpyHostToDevice);
}

void GPUSolver::freeGpuData() {
    // free memory on GPU
    cudaFree(phi_d_);
    cudaFree(phiNew_d_);
    cudaFree(phiAnalytic_d_);
    cudaFree(f_d_);
    cudaFree(xPosFace_d_);
    cudaFree(xNegFace_d_);
    cudaFree(yPosFace_d_);
    cudaFree(yNegFace_d_);
    cudaFree(zPosFace_d_);
    cudaFree(zNegFace_d_);
    cudaFree(sendBuffer_d_);
    cudaFree(recvBuffer_d_);
}