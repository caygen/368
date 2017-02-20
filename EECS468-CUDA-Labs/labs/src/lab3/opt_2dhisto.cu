#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histoKernel(uint32_t*, size_t, size_t, uint32_t*);
__global__ void opt_32to8Kernel(uint32_t*, uint8_t*, size_t);

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins, uint32_t* g_bins)
{
    /* This function should only contain a call to the GPU
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */
    histoKernel<<<INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80) / 1024 , 1024>>>(input, height, width, g_bins);
    opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH / 512, 512>>>(g_bins, bins, 1024);
    cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */

__global__ void histoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){
     int i = blockDim.x * blockIdx.x + threadIdx.x;
     if (i < 1024)
        bins[i] = 0;
     __syncthreads();
     int stride = blockDim.x * gridDim.x;
     while (i < 4096 * height)
     {
        if (i %  ((INPUT_WIDTH + 128) & 0xFFFFFF80) < width )
           atomicAdd( &(bins[input[i]]), 1 );
        i += stride;
     }
}

__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	output[idx] = (uint8_t)((input[idx] < UINT8_MAX) * input[idx]) + (input[idx] >= UINT8_MAX) * UINT8_MAX;

	__syncthreads();
}

void* AllocateDevice(size_t size){
	void* ret;
	cudaMalloc(&ret, size);
	return ret;
}

void CopyToDevice(void* D_device, void* D_host, size_t size){
	cudaMemcpy(D_device, D_host, size,
					cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* D_host, void* D_device, size_t size){
	cudaMemcpy(D_host, D_device, size,
					cudaMemcpyDeviceToHost);
}

void FreeDevice(void* D_device){
	cudaFree(D_device);
}
