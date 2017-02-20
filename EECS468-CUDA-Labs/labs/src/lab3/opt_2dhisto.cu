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


    //Creating 32_bit histogram in parallel
    //blockDim is 32*32 = 1024 threads per block
    //gridDim is (size of the input data) / (blockDim)

    histoKernel<<<INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80) / 1024 , 1024>>>(input, height, width, g_bins);

    //Converting 32_bit histogram to 8 bit
    opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH / 1024, 1024>>>(g_bins, bins, 1024);
    cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */

/*__global__ void histoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){
     int globalTid = blockDim.x * blockIdx.x + threadIdx.x;
     //__shared__ uint32_t* s_input = input;
     if (globalTid < 1024)
        bins[globalTid] = 0;
     __syncthreads();
     int stride = blockDim.x * gridDim.x;
     while (globalTid < 4096 * height)
     {
        if (globalTid %  ((INPUT_WIDTH + 128) & 0xFFFFFF80) < width )
           atomicAdd( &(bins[input[globalTid]]), 1 );
        globalTid += stride;
     }
}
*/
__global__ void histoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){

	int col = blockDim.x * blockIdx.x + threadIdx.x;
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int mask = (INPUT_WIDTH + 128) & 0xFFFFFF80;

	if (row == 0 && col < 1024) {
		bins[col] = 0;
	}

	int index;
	__syncthreads();
	if (row < height && col < width) {
		index = input[col + row * mask];
		if (bins[index] < 255)
			atomicAdd(&bins[index], 1);
	}

}
__global__ void opt_32to8Kernel(uint32_t *input, uint8_t* output, size_t length){
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	output[idx] = (uint8_t)((input[idx] < UINT8_MAX) * input[idx]) + (input[idx] >= UINT8_MAX) * UINT8_MAX;
	__syncthreads();
}

void* AllocateOnDevice(size_t size){
	void* ret;
	cudaMalloc(&ret, size);
	return ret;
}

void CopyToDevice(void* d_device, void* d_host, size_t size){
	cudaMemcpy(d_device, d_host, size,cudaMemcpyHostToDevice);
}

void CopyFromDevice(void* d_host, void* d_device, size_t size){
	cudaMemcpy(d_host, d_device, size,cudaMemcpyDeviceToHost);
}

void FreeCuda(void* d_space){
	cudaFree(d_space);
}
