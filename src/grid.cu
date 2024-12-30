#include "grid.cuh"
#include "math_solvers.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

constexpr double initPhi = 0.5;

// 3D local grid implementation
Grid::Grid(unsigned int NX, unsigned int NY, unsigned int NZ, 
        double dx, double dy, double dz, 
        unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ,
        unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ) :
    NX_(NX), NY_(NY), NZ_(NZ), dx_(dx), dy_(dy), dz_(dz) {
    // allocate data
    int err = initGrid(totalNX, totalNY, totalNZ, localStartX, localStartY, localStartZ);
    if (err != 0) {
        std::cout << "Error initializing grid!" << std::endl;
        exit(1);
    }

    // verify
    // err = verifyInit(totalNX, totalNY, totalNZ, localStartX, localStartY, localStartZ);
    // if (err != 0) {
    //     std::cout << "Error verifying grid!" << std::endl;
    //     std::cout << "Number of errors: " << err << std::endl;
    //     exit(1);
    // }
}

// factory
std::shared_ptr<Grid> Grid::createGrid(
        unsigned int NX, unsigned int NY, unsigned int NZ, 
        double dx, double dy, double dz, 
        unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ, 
        unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ) {
    return std::shared_ptr<Grid>(new Grid(NX, NY, NZ, dx, dy, dz, 
                totalNX, totalNY, totalNZ, localStartX, localStartY, localStartZ));
}

Grid::~Grid() {
    // free memory
    freeGrid();
}

// get the local grid size
unsigned int Grid::getNX() {
    return NX_;
}
unsigned int Grid::getNY() {
    return NY_;
}
unsigned int Grid::getNZ() {
    return NZ_;
}

// get the grid spacing
double Grid::getDX() {
    return dx_;
}
double Grid::getDY() {
    return dy_;
}
double Grid::getDZ() {
    return dz_;
}

// get data
double *Grid::getPhi() {
    return phi_;
}
double *Grid::getPhiNew() {
    return phiNew_;
}
double *Grid::getAnalyticPhi() {
    return phiAnalytic_;
}
double *Grid::getF() {
    return f_;
}

void Grid::swap() {
    std::swap(phi_, phiNew_);
}

// cuda kernel to initialize the grid
__global__ 
void initGridKernel(
        double *phi, double *phiNew, double *phiAnalytic, double *f,
        unsigned int NX, unsigned int NY, unsigned int NZ, 
        unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ,
        unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ,
        double dx, double dy, double dz) {
            
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < NX && j < NY && k < NZ) {
        double x = (i + localStartX) * dx;
        double y = (j + localStartY) * dy;
        double z = (k + localStartZ) * dz;
        double exactF = fExact(x, y, z);
        double exactPhi = phiExact(x, y, z);
        unsigned int index = i + j*NX + k*NX*NY;
        // global boundary conditions
        if (i + localStartX == 0 || i + localStartX == totalNX - 1 ||
                j + localStartY == 0 || j + localStartY == totalNY - 1 ||
                k + localStartZ == 0 || k + localStartZ == totalNZ - 1) {
            phi[index] = exactPhi;
            phiNew[index] = exactPhi;
        }
        else {
            phi[index] = initPhi;
            phiNew[index] = initPhi;
        }
        f[index] = exactF;
        phiAnalytic[index] = exactPhi;
    }
}

int Grid::initGrid(unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ, 
        unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ) {
    // alloc
    cudaHostAlloc((void**)&phi_, NX_*NY_*NZ_*sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&phiNew_, NX_*NY_*NZ_*sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&phiAnalytic_, NX_*NY_*NZ_*sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&f_, NX_*NY_*NZ_*sizeof(double), cudaHostAllocDefault);

    // init alloc for gpu fill kernel
    double *phi_d;
    double *phiNew_d;
    double *phiAnalytic_d;
    double *f_d;
    cudaMalloc((void**)&phi_d, NX_*NY_*NZ_*sizeof(double));
    cudaMalloc((void**)&phiNew_d, NX_*NY_*NZ_*sizeof(double));
    cudaMalloc((void**)&phiAnalytic_d, NX_*NY_*NZ_*sizeof(double));
    cudaMalloc((void**)&f_d, NX_*NY_*NZ_*sizeof(double));

    // init
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks(
            (NX_ + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (NY_ + threadsPerBlock.y - 1) / threadsPerBlock.y,
            (NZ_ + threadsPerBlock.z - 1) / threadsPerBlock.z
    );

    initGridKernel<<<numBlocks, threadsPerBlock>>>(
            phi_d, phiNew_d, phiAnalytic_d, f_d,
            NX_, NY_, NZ_,
            totalNX, totalNY, totalNZ,
            localStartX, localStartY, localStartZ,
            dx_, dy_, dz_
    );

    cudaDeviceSynchronize();

    // copy to cpu
    cudaMemcpy(phi_, phi_d, NX_*NY_*NZ_*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(phiNew_, phiNew_d, NX_*NY_*NZ_*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(phiAnalytic_, phiAnalytic_d, NX_*NY_*NZ_*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(f_, f_d, NX_*NY_*NZ_*sizeof(double), cudaMemcpyDeviceToHost);

    // free gpu memory
    cudaFree(phi_d);
    cudaFree(phiNew_d);
    cudaFree(phiAnalytic_d);
    cudaFree(f_d);

    if (cudaPeekAtLastError() != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << std::endl;
        return 1;
    }

    return 0;
}

void Grid::freeGrid() {
    cudaFreeHost(phi_);
    cudaFreeHost(phiNew_);
    cudaFreeHost(phiAnalytic_);
    cudaFreeHost(f_);

    phi_ = nullptr; 
    phiNew_ = nullptr;
    phiAnalytic_ = nullptr;
    f_ = nullptr;
}

int Grid::verifyInit(unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ, 
        unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ) {
    
    int err = 0;
    for (unsigned int i = 0; i < NX_; i++) {
        double x = (i + localStartX)*dx_;
        for (unsigned int j = 0; j < NY_; j++) {
            double y = (j + localStartY)*dy_;
            for (unsigned int k = 0; k < NZ_; k++) {
                double z = (k + localStartZ)*dz_;

                double exactPhi = phiExactCPU(x, y, z);
                double exactF = fExactCPU(x, y, z);

                unsigned int index = Index(i, j, k);
                if (i + localStartX == 0 || i + localStartX == totalNX - 1 ||
                        j + localStartY == 0 || j + localStartY == totalNY - 1 ||
                        k + localStartZ == 0 || k + localStartZ == totalNZ - 1) {
                    if (fabs(phi_[index] - exactPhi) > 1e-6) {
                        err++;
                    }
                    if (fabs(phiNew_[index] - exactPhi) > 1e-6) {
                        err++;
                    }
                    if (fabs(phiAnalytic_[index] - exactPhi) > 1e-6) {
                        err++;
                    }
                    if (fabs(f_[index] - exactF) > 1e-6) {
                        err++;
                    }
                }
                else {
                    if (fabs(phi_[index] - initPhi) > 1e-6) {
                        err++;
                    }
                    if (fabs(phiNew_[index] - initPhi) > 1e-6) {
                        err++;
                    }
                    if (fabs(phiAnalytic_[index] - exactPhi) > 1e-6) {
                        err++;
                    }
                    if (fabs(f_[index] - exactF) > 1e-6) {
                        err++;
                    }
                }
            }
        }
    }

    return err;
}

// print out every phi[i,j,k], i, j, k
void Grid::printPhi() {
    for (unsigned int i = 0; i < NX_; i++) {
        for (unsigned int j = 0; j < NY_; j++) {
            for (unsigned int k = 0; k < NZ_; k++) {
                unsigned int index = Index(i, j, k);
                printf("%f, %d,%d,%d ", phi_[index], i, j, k);
            }
            printf("\n");
        }
        printf("\n");
    }
}
