#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"
#include "opt_2dhisto.h"

void opt_2dhisto(int size, uint32_t* dinput, uint32_t* dbins)
{
    dim3 dimBlock(1024);
    dim3 dimGrid(1);
    HistoKernel<<<dimGrid, dimBlock>>>(size, dinput, dbins); 
       /* This function should only contain a call to the GPU 
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */

}

void parallel32to8copy(uint32_t* dbins, uint8_t* dout)
{
    dim3 dimBlock(1024);
    dim3 dimGrid(512);
    CopyKernel<<<dimGrid, dimBlock>>>(dbins, dout); 

/* Include below the implementation of any other functions you need */
}

__global__ void CopyKernel(uint32_t* dbins, uint8_t* dout)
{
    const int globaltid = blockIdx.x * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * gridDim.x;

    const int BIN_COUNT = HISTO_HEIGHT * HISTO_WIDTH;

    for (int i = globaltid; i < BIN_COUNT; i += numThreads)
    {
        dout[i] = dbins[i] <= 255 ? dbins[i] : 255;
    
    }
}

void setUp(void* dinput,void* dout, void* dbins, unsigned int size, void** input)
{

    cudaMalloc((void**)&dout, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    printf("dout malloc complete\n");
    cudaMalloc((void**)&dinput, size);
    printf("dinput malloc complete\n");
    cudaMalloc((void**)&dbins, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
    printf("dbins malloc complete\n");
    cudaMemset(dbins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
    printf("dbins memset complete\n");
    cudaMemset(dout, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t));
    printf("dout memset complete\n");

    cudaMemcpy(dinput, *input, size, cudaMemcpyHostToDevice); 
    printf("memcpy complete\n");
    

}
    /* End of setup code */

void tearDown(void* kernel_bins,void* dout, void* dbins, void* dinput) 
{
    /* Include your teardown code below (temporary variables, function calls, etc.) */

    cudaThreadSynchronize();
    cudaMemcpy(kernel_bins,dout, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint8_t), cudaMemcpyDeviceToHost); 
    cudaFree(dinput);
    cudaFree(dbins);
    cudaFree(dout);
}

__global__ void HistoKernel(int size, uint32_t* dinput, uint32_t* dbins) 
{
    const int BIN_COUNT = HISTO_HEIGHT * HISTO_WIDTH; 
    const int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    
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
	// unsigned int curr = dbins[pos];
        atomicAdd(dbins + pos, s_Hist[pos]);
        // if (curr > dbins[pos]) dbins[pos] = 255;
    }

}
