#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#define N 600
#define DIMS 2
#define epsilon 0.1
#define ss 1
#define blocksize 16
#define BLK (blocksize*blocksize)
#define dimGrid (N / BLK + 1)

__device__ float distance(float *a, float *b) {
    int i;
    float d = 0.0, t;
    for(i=0;i<DIMS;i++) {
        t = a[i] - b[i];
        d += t*t;
    }
    return d;
}

__device__ void meanshift(float *x, float *y, float *result) {
    int j, i;
    float sum1[DIMS], sum2 = 0.0, dis, k;
    for (i=0;i<DIMS;i++) sum1[i] = 0.0;
    for(j=0;j<N;j++) {
        dis = distance(y,x+DIMS*j);
        if (dis > ss*ss) continue;
        k = exp( - dis / (2*ss) );
        sum2 += k;
        for(i=0;i<DIMS;i++) sum1[i] += k * x[DIMS*j+i];
    }
    for(i=0;i<DIMS;i++) result[i] = sum1[i] / sum2;
}

__global__ void msfunction(float *x, float *y) {
    __shared__ float ys[BLK*DIMS], temp[BLK*DIMS], m[BLK];
    int h, j = threadIdx.x * blockDim.x  + threadIdx.y;
    int i = blockIdx.x * BLK + j;
    if (i < N) {
        memcpy(ys+DIMS*j, x+DIMS*i, DIMS*sizeof(float));
        do {
            meanshift(x, ys+DIMS*j, temp+DIMS*j);
            m[j] = sqrt(distance(temp+DIMS*j, ys+DIMS*j));
            memcpy(ys+DIMS*j, temp+DIMS*j, DIMS*sizeof(float));
        } while (m[j]>epsilon);
        for(h=0;h<DIMS;h++)
            y[DIMS*i+h] = ys[DIMS*j+h];
    }
}

int main() {
    dim3 dimBlock( blocksize, blocksize );
    int i, j, errors = 0;
    float *x, *y, *xd, *yd, msecs, temp;
    clock_t start, end;
    x = (float *)malloc(DIMS*N*sizeof(float));
    y = (float *)malloc(DIMS*N*sizeof(float));
    FILE *data = fopen("data.txt", "r"), *output = fopen("meanshift.txt", "w"), *check = fopen("results.txt", "r");
    for (i=0;i<DIMS*N;i++)
        fscanf(data, "%f", &x[i]);
    fclose(data);
    cudaMalloc(&xd, N*DIMS*sizeof(float));
    cudaMalloc(&yd, N*DIMS*sizeof(float));
    cudaMemcpy(xd, x, N*DIMS*sizeof(float),cudaMemcpyHostToDevice);
    start = clock();
    msfunction<<<dimGrid,dimBlock>>>(xd, yd);
    cudaThreadSynchronize();
    end = clock();
    msecs = (float)1000*(end - start) / CLOCKS_PER_SEC;
    cudaMemcpy(y, yd, N*DIMS*sizeof(float),cudaMemcpyDeviceToHost);
    for (i=0;i<N;i++)
        for (j=0;j<DIMS;j++) {
            if (j==DIMS-1) fprintf(output, "%f\n", y[DIMS*i+j]);
            else fprintf(output, "%f ", y[DIMS*i+j]);
        }
    for (i=0;i<N*DIMS;i++) {
        fscanf(check, "%f", &temp);
        if ( fabs( temp - y[i] ) > 0.1) errors++;
    }
    printf("Time is %f msecs\n", msecs);
    printf("detected %d errors, %.2f %% of the total values\n", errors, (float)100*errors/(N*DIMS));
    fclose(output);
    fclose(check);
    cudaFree(xd);
    cudaFree(yd);
    free(x);
    free(y);
}