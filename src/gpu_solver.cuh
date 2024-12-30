#include <cuda.h>
#include "grid.cuh"

// Helper function to check CUDA errors
#define cudaCheckError() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// neighbor struct
struct Neighbors {
    double xPos;
    double xNeg;
    double yPos;
    double yNeg;
    double zPos;
    double zNeg;
};

// face struct
struct Face {
    double *start;
    unsigned int dim1;
    unsigned int dim2;
    unsigned int inc1;
    unsigned int inc2;
    unsigned int selfInc;
};

// gpu based solver for 3D Poisson equation
class GPUSolver {
public:
    // constructor
    GPUSolver(std::shared_ptr<Grid> grid);

    // destructor
    ~GPUSolver();

    // solve the Poisson equation, update the new grid
    void solveRB(double &error);
    void solveJacobi(double &error);

    // external funcs
    void copyResultsToCpu(double *phiNew);
    void copyResultsToGpu();
    double *getSendBuffer();
    double *getRecvBuffer();
    void copyFacesToSendBuffer(int (&neighborRanks)[6]);
    void copyRecvBufferToFaces(int (&neighborRanks)[6]);
    void copyBufferToCpu(double *testBuffer, double *buffer_d, unsigned int size);
    void sync();
    double *getPhiGpu();

private:
    // local grid
    std::shared_ptr<Grid> grid_;

    // faces
    Face *xPosFace_d_;
    Face *xNegFace_d_;
    Face *yPosFace_d_;
    Face *yNegFace_d_;
    Face *zPosFace_d_;
    Face *zNegFace_d_;
    unsigned int faceDims_[6];

    // send and receive buffers
    double *sendBuffer_d_;
    double *recvBuffer_d_;

    // func data
    double *phi_d_;
    double *phiNew_d_;
    double *phiAnalytic_d_;
    double *f_d_;

    // helper funcs
    void initGpuData();
    void freeGpuData();
    void createPlane(double *first, 
        unsigned int dim1, unsigned int dim2, unsigned int inc1, unsigned int inc2, 
        unsigned int selfInc, Face *plane);
    void swapFaces(double *currPhi);
};