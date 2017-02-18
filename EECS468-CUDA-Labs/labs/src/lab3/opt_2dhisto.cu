#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

void opt_2dhisto(int size, uint32_t* dinput, uint8_t* dbins)
{
    dim3 dimBlock(1024);
    dim3 dimGrid(512);
    HistoKernel<<<dimGrid, dimBlock>>>(size, dinput, dbins); 
       /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

}


/* Include below the implementation of any other functions you need */


void setUp(void* dinput, void* dbins, int size, uint32_t** input)
{

    cudaMalloc((void**)&dinput, size);
    cudaMalloc((void**)&dbins, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    cudaMemset(dbins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    cudaMemcpy(dinput, *input, size, cudaMemcpyHostToDevice); 

}
    /* End of setup code */

void tearDown(void* kernel_bins, void* dbins, void* dinput) 
{
    /* Include your teardown code below (temporary variables, function calls, etc.) */

    cudaThreadSynchronize();
    cudaMemcpy(kernel_bins, dbins, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t), cudaMemcpyDeviceToHost); 
    cudaFree(dinput);
    cudaFree(dbins);
}

__global__ void HistoKernel(int size, uint32_t* dinput, uint8_t* dbins) 
{
    const int BIN_COUNT = HISTO_HEIGHT * HISTO_WIDTH; 
    const int globalTid = blockIdx.x + blockDim.x + threadIdx.x;
    
    const int numThreads = blockDim.x * gridDim.x;

    __shared__ unsigned int s_Hist[BIN_COUNT]; 

    for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
    {
       s_Hist[pos] = 0;
    }
    __syncthreads();
 
    for (int pos = globalTid; pos < size; pos += numThreads)
    {
        uint32_t curr_Data = dinput[pos];

        atomicAdd(s_Hist + curr_Data, 1);
    }

    __syncthreads();

    for (int pos = threadIdx.x; pos < BIN_COUNT; pos += blockDim.x)
    {
	uint8_t curr = dbins[pos];
        atomicAdd((int*)dbins + pos, s_Hist[pos]);
        if (curr > dbins[pos]) dbins[pos] = 255;
    }

}
