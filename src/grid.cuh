#pragma once

#include <cuda.h>
#include <memory>

// the 3D local grid class for the 3D poisson solver
class Grid {
protected:
    // default constructor
    Grid(unsigned int NX, 
        unsigned int NY, 
        unsigned int NZ, 
        double dx,
        double dy,
        double dz,
        unsigned int totalNX,
        unsigned int totalNY,
        unsigned int totalNZ,
        unsigned int localStartX,
        unsigned int localStartY, 
        unsigned int localStartZ
    );
    
public:
    // factory constructor
    static std::shared_ptr<Grid> createGrid(
        unsigned int NX, 
        unsigned int NY, 
        unsigned int NZ, 
        double dx,
        double dy,
        double dz,
        unsigned int totalNX,
        unsigned int totalNY,
        unsigned int totalNZ,
        unsigned int localStartX,
        unsigned int localStartY, 
        unsigned int localStartZ
    );

    // destructor
    ~Grid();

    // get the local grid size
    unsigned int getNX();
    unsigned int getNY();
    unsigned int getNZ();

    // get the grid spacing
    double getDX();
    double getDY();
    double getDZ();

    // get data
    double *getPhi();
    double *getPhiNew();
    double *getAnalyticPhi();
    double *getF();

    // get index
    inline unsigned int Index(unsigned int i, unsigned int j, unsigned int k) {
        return i + j * NX_ + k * NX_ * NY_;
    }

    // swap
    void swap();

    // test funcs
    int verifyInit(unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ, 
            unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ);
    void printPhi();

private:
    // local grid size
    unsigned int NX_;
    unsigned int NY_;
    unsigned int NZ_;

    // grid spacing
    double dx_;
    double dy_;
    double dz_;

    // func data
    double *phi_;
    double *phiNew_;
    double *phiAnalytic_;
    double *f_;

    // helper funcs
    int initGrid(unsigned int totalNX, unsigned int totalNY, unsigned int totalNZ, 
            unsigned int localStartX, unsigned int localStartY, unsigned int localStartZ);
    void freeGrid();
    
};