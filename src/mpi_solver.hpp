#pragma once

#include "grid.cuh"
#include "gpu_solver.cuh"
#include "math_solvers.cuh"
#include <mpi.h>
#include <memory>

// the MPI solver class to parallelize the 3D poisson solver
class MPISolver {
public:
    // constructor
    MPISolver(unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ,
        double dx, double dy, double dz);
        
    // destructor
    ~MPISolver();

    // solve the poisson equation
    void solve(double tolerance, unsigned int maxIterations, bool warmingUp);

private:
    // mpi data
    int rank_;
    int numProcs_;
    MPI_Comm cartComm_;
    int coords_[3];

    // neighbors
    int posX_;
    int negX_;
    int posY_;
    int negY_;
    int posZ_;
    int negZ_;

    // whole grid size
    unsigned int totalNX_;
    unsigned int totalNY_;
    unsigned int totalNZ_;

    // local grid
    std::shared_ptr<Grid> grid_;

    unsigned int localStartX_;
    unsigned int localStartY_;
    unsigned int localStartZ_;

    // helper funcs
    void decomposeGrid(unsigned int &localNX, unsigned int &localNY, unsigned int &localNZ);
    void exchange(MPI_Request *reqs, double *sendBuffer_d, double *recvBuffer_d);
    void exchangeCPU();

    // self cpu solver
    void solveCPU(double &error);
};