#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void __global__ add(const double *x, const double *y, double *z) {
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

int main(void) {
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*)malloc(M);
    double *h_y = (double*)malloc(M);
    double *h_z = (double*)malloc(M);

    for(int n = 0; n < N; n++) {
        h_x[n] = (double)n / 100.0f;
        h_y[n] = (double)n / 100.0f;
    }

    double *d_x, *d_y, *d_z;
    cudaMalloc((void **)&d_x, M);
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);

    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128;
    const int grid_size = N / block_size;

    for(int i = 0; i < 1000; i++) {
        add<<<grid_size, block_size>>>(d_x, d_y, d_z);
    }

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);

    free(h_x);
    free(h_y);
    free(h_z);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
