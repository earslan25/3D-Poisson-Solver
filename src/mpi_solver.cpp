#include "mpi_solver.hpp"
#include <iostream>
#include <cmath>
#include <mpi.h>
#include "mpi.h"
#include <unistd.h> 


// MPI solver with domain decomposition for the 3D Poisson equation
// jacobi iteration is used
MPISolver::MPISolver(unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ,
        double dx, double dy, double dz) :
    totalNX_(totalNX), totalNY_(totalNY), totalNZ_(totalNZ), grid_(nullptr) {

    // init grid based on global domain size
    unsigned int localNX = 0;
    unsigned int localNY = 0;
    unsigned int localNZ = 0;
    decomposeGrid(localNX, localNY, localNZ);

    // print info
    printf("rank %d of %d\n", rank_, numProcs_);
    printf("local start: %d %d %d\n", localStartX_, localStartY_, localStartZ_);

    grid_ = Grid::createGrid(localNX, localNY, localNZ, dx, dy, dz, 
            totalNX_, totalNY_, totalNZ_, 
            localStartX_, localStartY_, localStartZ_);
}

MPISolver::~MPISolver() {

}

// solve the Poisson equation
void MPISolver::solve(double tolerance, unsigned int maxIterations, bool warmingUp) {
    // solver data
    double error = 0.0;
    double residual = 0.0;
    unsigned int iter = 0;
    double *phiAnalytic = grid_->getAnalyticPhi();
    double *phi = grid_->getPhi();

    // gpu data
    GPUSolver *gpuSolver = new GPUSolver(grid_);
    double *sendBuffer_d = gpuSolver->getSendBuffer();
    double *recvBuffer_d = gpuSolver->getRecvBuffer();

    // MPI data
    MPI_Request *reqs = new MPI_Request[12];
    MPI_Status *stats = new MPI_Status[12];
    int neighborRanks[6] = {posX_, negX_, posY_, negY_, posZ_, negZ_};

    // start timer
    double startTime = MPI_Wtime();

    //solve loop
    for (iter = 0; iter < maxIterations; iter++) {
        // prepare send buffers
        gpuSolver->copyFacesToSendBuffer(neighborRanks);

        // exchange
        exchange(reqs, sendBuffer_d, recvBuffer_d);
        // exchangeCPU();

        // wait for all to finish
        MPI_Waitall(12, reqs, stats);

        // update faces from recv buffer
        gpuSolver->copyRecvBufferToFaces(neighborRanks);

        double currError = 0.0;
        // solve
        gpuSolver->solveRB(currError);
        // gpuSolver->solveJacobi(currError);

        // get errors from all processes
        MPI_Allreduce(MPI_IN_PLACE, &currError, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // calculate residual and error
        residual = sqrt(currError);
        error = currError / ((totalNX_ -2) * (totalNY_ - 2) * (totalNZ_ - 2));

        if (error < tolerance) {
            break;
        }

        if (rank_ == 0) {
            if (iter % 100 == 0) {
                std::cout << "Iteration: " << iter << " Residual: " << residual << std::endl;
                std::cout << "Error: " << error << std::endl;
            }
        }
    }

    // stop timer
    double endTime = MPI_Wtime();
    double totalTime = endTime - startTime;

    // print results if not warmup
    if (rank_ == 0 && !warmingUp) {
        std::cout << "Solver finished in " << iter << " iterations with residual " << residual << 
            " and error " << error << std::endl;
        std::cout << "Time: " << totalTime << std::endl;
    }

    // clean up
    delete gpuSolver;
    delete[] reqs;
    delete[] stats;
}

void MPISolver::decomposeGrid(unsigned int &localNX, 
        unsigned int &localNY, unsigned int &localNZ) {
    // MPI cart
    int dims[3] = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs_);
    MPI_Dims_create(numProcs_, 3, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cartComm_);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Cart_coords(cartComm_, rank_, 3, coords_);
    
    // set neighbors
    MPI_Cart_shift(cartComm_, 0, 1, &negX_, &posX_);
    MPI_Cart_shift(cartComm_, 1, 1, &negY_, &posY_);
    MPI_Cart_shift(cartComm_, 2, 1, &negZ_, &posZ_);

    // decompose the grid
    localNX = totalNX_ / dims[0];
    localNY = totalNY_ / dims[1];
    localNZ = totalNZ_ / dims[2];

    localStartX_ = localNX * coords_[0];
    localStartY_ = localNY * coords_[1];
    localStartZ_ = localNZ * coords_[2];
}

// halo exchange
void MPISolver::exchange(MPI_Request *reqs, double *sendBuffer_d, double *recvBuffer_d) {
    // use cuda aware MPI to exchange data
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    // send and receive
    unsigned int start = 0;

    // x direction
    MPI_Isend(sendBuffer_d, NY * NZ, MPI_DOUBLE, posX_, 0, cartComm_, &reqs[0]);
    MPI_Irecv(recvBuffer_d, NY * NZ, MPI_DOUBLE, posX_, 1, cartComm_, &reqs[1]);
    start += NY * NZ;

    MPI_Isend(sendBuffer_d + start, NY * NZ, MPI_DOUBLE, negX_, 1, cartComm_, &reqs[2]);
    MPI_Irecv(recvBuffer_d + start, NY * NZ, MPI_DOUBLE, negX_, 0, cartComm_, &reqs[3]);
    start += NY * NZ;

    // y direction
    MPI_Isend(sendBuffer_d + start, NX * NZ, MPI_DOUBLE, posY_, 2, cartComm_, &reqs[4]);
    MPI_Irecv(recvBuffer_d + start, NX * NZ, MPI_DOUBLE, posY_, 3, cartComm_, &reqs[5]);
    start += NX * NZ;

    MPI_Isend(sendBuffer_d + start, NX * NZ, MPI_DOUBLE, negY_, 3, cartComm_, &reqs[6]);
    MPI_Irecv(recvBuffer_d + start, NX * NZ, MPI_DOUBLE, negY_, 2, cartComm_, &reqs[7]);
    start += NX * NZ;

    // z direction
    MPI_Isend(sendBuffer_d + start, NX * NY, MPI_DOUBLE, posZ_, 4, cartComm_, &reqs[8]);
    MPI_Irecv(recvBuffer_d + start, NX * NY, MPI_DOUBLE, posZ_, 5, cartComm_, &reqs[9]);
    start += NX * NY;

    MPI_Isend(sendBuffer_d + start, NX * NY, MPI_DOUBLE, negZ_, 5, cartComm_, &reqs[10]);
    MPI_Irecv(recvBuffer_d + start, NX * NY, MPI_DOUBLE, negZ_, 4, cartComm_, &reqs[11]);
}

/* not tested! */
void MPISolver::exchangeCPU() {
    MPI_Request reqs[12];
    MPI_Status stats[12];

    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();
    double *phi = grid_->getPhi();

    // send buffers
    double *bufferX1 = new double[NY * NZ];  
    double *bufferX2 = new double[NY * NZ];
    double *bufferY1 = new double[NX * NZ];
    double *bufferY2 = new double[NX * NZ];
    double *bufferZ1 = new double[NX * NY];
    double *bufferZ2 = new double[NX * NY];

    // receive buffers
    double *bufferX1_r = new double[NY * NZ];
    double *bufferX2_r = new double[NY * NZ];
    double *bufferY1_r = new double[NX * NZ];
    double *bufferY2_r = new double[NX * NZ];
    double *bufferZ1_r = new double[NX * NY];
    double *bufferZ2_r = new double[NX * NY];

    // sends
    for (unsigned int i = 0; i < NY; i++) {
        for (unsigned int j = 0; j < NZ; j++) {
            bufferX1[i * NZ + j] = phi[1 + i * NZ + j * NX];
            bufferX2[i * NZ + j] = phi[NX - 2 + i * NZ + j * NX];
        }
    }

    for (unsigned int i = 0; i < NX; i++) {
        for (unsigned int j = 0; j < NZ; j++) {
            bufferY1[i * NZ + j] = phi[i + NX + j * NX * NY];
            bufferY2[i * NZ + j] = phi[i + (NY - 2) * NX + j * NX * NY];
        }
    }

    for (unsigned int i = 0; i < NX; i++) {
        for (unsigned int j = 0; j < NY; j++) {
            bufferZ1[i * NY + j] = phi[i + j * NX + NX * NY];
            bufferZ2[i * NY + j] = phi[i + j * NX + (NZ - 2) * NX * NY];
        }
    }

    unsigned int count = 0;

    // send and receive from each direction
    if (posX_ != -2) {
        MPI_Isend(bufferX1, NY * NZ, MPI_DOUBLE, posX_, 0, cartComm_, &reqs[count]);
        MPI_Irecv(bufferX1_r, NY * NZ, MPI_DOUBLE, posX_, 1, cartComm_, &reqs[count + 1]);
        count += 2;
    }
    if (negX_ != -2) {
        MPI_Isend(bufferX2, NY * NZ, MPI_DOUBLE, negX_, 1, cartComm_, &reqs[count]);
        MPI_Irecv(bufferX2_r, NY * NZ, MPI_DOUBLE, negX_, 0, cartComm_, &reqs[count + 1]);
        count += 2;
    }
    if (posY_ != -2) {
        MPI_Isend(bufferY1, NX * NZ, MPI_DOUBLE, posY_, 2, cartComm_, &reqs[count]);
        MPI_Irecv(bufferY1_r, NX * NZ, MPI_DOUBLE, posY_, 3, cartComm_, &reqs[count + 1]);
        count += 2;
    }
    if (negY_ != -2) {
        MPI_Isend(bufferY2, NX * NZ, MPI_DOUBLE, negY_, 3, cartComm_, &reqs[count]);
        MPI_Irecv(bufferY2_r, NX * NZ, MPI_DOUBLE, negY_, 2, cartComm_, &reqs[count + 1]);
        count += 2;
    }
    if (posZ_ != -2) {
        MPI_Isend(bufferZ1, NX * NY, MPI_DOUBLE, posZ_, 4, cartComm_, &reqs[count]);
        MPI_Irecv(bufferZ1_r, NX * NY, MPI_DOUBLE, posZ_, 5, cartComm_, &reqs[count + 1]);
        count += 2;
    }
    if (negZ_ != -2) {
        MPI_Isend(bufferZ2, NX * NY, MPI_DOUBLE, negZ_, 5, cartComm_, &reqs[count]);
        MPI_Irecv(bufferZ2_r, NX * NY, MPI_DOUBLE, negZ_, 4, cartComm_, &reqs[count + 1]); 
        count += 2;
    }

    // wait for all to finish
    MPI_Waitall(count, reqs, stats);

    // unpack
    for (unsigned int i = 0; i < NY; i++) {
        for (unsigned int j = 0; j < NZ; j++) {
            phi[NX * (NY - 1) + i * NZ + j] = bufferX1_r[i * NZ + j];
            phi[i * NZ + j] = bufferX2_r[i * NZ + j];
        }
    }

    for (unsigned int i = 0; i < NX; i++) {
        for (unsigned int j = 0; j < NZ; j++) {
            phi[i * NZ + NY - 1 + j * NX] = bufferY1_r[i * NZ + j];
            phi[i * NZ + j * NX] = bufferY2_r[i * NZ + j];
        }
    }

    for (unsigned int i = 0; i < NX; i++) {
        for (unsigned int j = 0; j < NY; j++) {
            phi[i * NY + j + (NZ - 1) * NX * NY] = bufferZ1_r[i * NY + j];
            phi[i * NY + j] = bufferZ2_r[i * NY + j];
        }
    }

    delete[] bufferX1;
    delete[] bufferX2;
    delete[] bufferY1;
    delete[] bufferY2;
    delete[] bufferZ1;
    delete[] bufferZ2;

    delete[] bufferX1_r;
    delete[] bufferX2_r;
    delete[] bufferY1_r;
    delete[] bufferY2_r;
    delete[] bufferZ1_r;
    delete[] bufferZ2_r;
}

void MPISolver::solveCPU(double &error) {
    unsigned int NX = grid_->getNX();
    unsigned int NY = grid_->getNY();
    unsigned int NZ = grid_->getNZ();

    double dx = grid_->getDX();
    double dy = grid_->getDY();
    double dz = grid_->getDZ();

    double *phiOld = grid_->getPhi();
    double *phiNew = grid_->getPhiNew();
    double *f = grid_->getF();
    double *phiAnalytic = grid_->getAnalyticPhi();

    // solve with jacobi + omp
    #pragma omp parallel for
    for (unsigned int i = 1; i < NX - 1; i++) {
        #pragma omp parallel for
        for (unsigned int j = 1; j < NY - 1; j++) {
            #pragma omp parallel for reduction(+:error)
            for (unsigned int k = 1; k < NZ - 1; k++) {
                // index = i + j * NX + k * NX * NY
                unsigned int idx = grid_->Index(i, j, k);
                phiNew[idx] = (phiOld[idx + 1] + phiOld[idx - 1] + 
                        phiOld[idx + NX] + phiOld[idx - NX] + 
                        phiOld[idx + NX * NY] + phiOld[idx - NX * NY] - 
                        f[idx]*dx*dy*dz) / 6.0;
                error += sqrErrCPU(phiNew[idx], phiAnalytic[idx]);
            }
        }
    }

    // update phi
    grid_->swap();
}