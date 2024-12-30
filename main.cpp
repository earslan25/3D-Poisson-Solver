#include "src/mpi_solver.hpp"
#include <mpi.h>
#include <memory>
#include <cmath>
#include <iostream>


constexpr unsigned int WARMUP = 3;
constexpr unsigned int TX = 80;
constexpr unsigned int TY = 80;
constexpr unsigned int TZ = 40;
constexpr double DX = 1.0 / (2.0*(double)TX);
constexpr double DY = 1.0 / (2.0*(double)TY);
constexpr double DZ = 1.0 / (2.0*(double)TZ);
constexpr double TOL = 1e-5;
constexpr unsigned int MAX_ITER = 3000;


int main(int argc, char **argv) {
    // init MPI
    MPI_Init(&argc, &argv);

    MPISolver *mpiSolver = new MPISolver(TX, TY, TZ, DX, DY, DZ);

    bool warmingUp = false;

    // solve
    mpiSolver->solve(TOL, MAX_ITER, warmingUp);

    MPI_Finalize();
    
    return 0;
}
