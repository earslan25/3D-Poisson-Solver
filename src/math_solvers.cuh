#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

// math solvers
// gradients, exact functions etc

// constants
__device__ constexpr double PI = 3.1415926;
constexpr unsigned int n = 2;
constexpr unsigned int m = 4;
constexpr unsigned int k = 8;

/* GPU */

// phi = sin(n*pi*x)*cos(m*pi*y)*sin(k*pi*z)
// sin(pi*x*y*z)
// phi = 6
__device__ __forceinline__
double phiExact(double x, double y, double z) {
    // return sin(n*PI*x) * cos(m*PI*y) * sin(k*PI*z);
    return sin(PI*x*y*z);
    // return 6;
}

// f = -n^2*pi^2*phi - m^2*pi^2*phi - k^2*pi^2*phi
// f = -pi^2*(x^2 + y^2 + z^2)*sin(pi*x*y*z)
// f = x^2 + y^2 + z^2
__device__ __forceinline__
double fExact(double x, double y, double z) {
    // return -PI*PI * (n*n + m*m + k*k) * phiExact(x, y, z);
    return -PI*PI * (x*x + y*y + z*z) * sin(PI*x*y*z);
    // return x*x + y*y + z*z;
}

// square error
__device__ __forceinline__
double sqrErr(double approxVal, double trueVal) {
    double diff = trueVal - approxVal;
    return diff * diff;
}

// numerical 2nd partial derivative of phi
// __device__ __forceinline__
// double pd2phi(double prev, double curr, double next, double h2) {
//     return (prev + next - h2*curr) * 0.5;
// }

/* CPU */

// phi = sin(n*pi*x)*cos(m*pi*y)*sin(k*pi*z)
// phi = sin(pi*x*y*z)
// phi = 6
inline double phiExactCPU(double x, double y, double z) {
    // return sin(n*PI*x) * cos(m*PI*y) * sin(k*PI*z);
    return sin(PI*x*y*z);
    // return 6;
}

// f = -n^2*pi^2*phi - m^2*pi^2*phi - k^2*pi^2*phi
// f = -pi^2*(x^2 + y^2 + z^2)*sin(pi*x*y*z)
// f = x^2 + y^2 + z^2
inline double fExactCPU(double x, double y, double z) {
    // return -PI*PI * (n*n + m*m + k*k) * phiExactCPU(x, y, z);
    return -PI*PI * (x*x + y*y + z*z) * sin(PI*x*y*z);
    // return x*x + y*y + z*z;
}

// square error
inline double sqrErrCPU(double approxVal, double trueVal) {
    double diff = trueVal - approxVal;
    return diff * diff;
}

// numerical 2nd partial derivative of phi
// inline double pd2phiCPU(double prev, double curr, double next, double h2) {
//     return (prev + next - h2*curr) * 0.5;
// }