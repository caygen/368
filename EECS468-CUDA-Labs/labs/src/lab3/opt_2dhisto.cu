#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void histoKernel(uint32_t*, size_t, size_t, uint32_t*);
__global__ void opt_32to8Kernel(uint32_t*, uint8_t*, size_t);


__global__ void histogram(unsigned int *input, unsigned int *bins,unsigned int num_elements,unsigned int num_bins);
__global__ void saturate(unsigned int *bins, unsigned int num_bins);

#define NUM_BINS = HISTO_HEIGHT * HISTO_WIDTH * sizeof(uint32_t);

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins, uint32_t* g_bins)
{
    /* This function should only contain a call to the GPU
       histogramming kernel. Any memory allocations and
       transfers must be done outside this function */


    //Creating 32_bit histogram in parallel
    //blockDim is 32*32 = 1024 threads per block
    //gridDim is (size of the input data) / (blockDim)

  //  histoKernel<<<INPUT_HEIGHT * ((INPUT_WIDTH + 128) & 0xFFFFFF80) / 1024 , 1024>>>(input, height, width, g_bins);

    //Converting 32_bit histogram to 8 bit
  //  opt_32to8Kernel<<<HISTO_HEIGHT * HISTO_WIDTH / 1024, 1024>>>(g_bins, bins, 1024);


  int dimGrid = HISTO_HEIGHT * HISTO_WIDTH / 1024;
  int dimBlock = 1024;
  histogram<<<dimGrid, dimBlock>>>(input, deviceBins, height*width, NUM_BINS);
  saturate<<<dimGrid, dimBlock>>>(deviceBins, NUM_BINS);
    cudaThreadSynchronize();
}

/* Include below the implementation of any other functions you need */

__global__ void histogram(unsigned int *input, unsigned int *bins,unsigned int num_elements,unsigned int num_bins) {
	//@@ privitization technique

	__shared__ unsigned int histo_private[NUM_BINS];

	int i = threadIdx.x + blockIdx.x * blockDim.x; // global thread id
	// total number of threads
	int stride = blockDim.x * gridDim.x;

	if (threadIdx.x < num_bins) {
		histo_private[threadIdx.x] = 0;
	}

	__syncthreads();

	// compute block's histogram

	while (i < num_elements) {
		int temp = input[i];
		atomicAdd(&(histo_private[temp]), 1);
		i += stride;
	}
	// wait for all other threads in the block to finish
	__syncthreads();

	// store to global histogram

	if (threadIdx.x < num_bins) {
		//int t = histo_private[threadIdx.x];
		atomicAdd(&(bins[threadIdx.x]), histo_private[threadIdx.x]);
	}

	/*
	for (int pos = threadIdx.x; pos < NUM_BINS; pos += blockDim.x) {
		histo_private[pos] = 0;
	}
	__syncthreads();
	for (int pos = i; pos < num_elements; pos += stride) {
		atomicAdd(&(histo_private[input[i]]), 1);
	}
	__syncthreads();
	for (int pos = threadIdx; pos < NUM_BINS; pos += blockDim.x) {
		atomicAdd(&(bins[threadIdx.x]), histo_private[threadIdx.x]);
	}
	*/

	/*
	histo_private[threadIdx.x] = 0;
	__syncthreads();
	while (i < num_elements) {
		atomicAdd(&histo_private[input[i]], 1);
		i += stride;
	}
	__syncthreads();
	atomicAdd(&bins[threadIdx.x], histo_private[threadIdx.x]);
	*/
}

__global__ void saturate(unsigned int *bins, unsigned int num_bins) {
	//@@ counters are saturated at 127

	int i = threadIdx.x + blockIdx.x * blockDim.x;

	if (i < num_bins) {
		if (bins[i] > 127) { // || bins[i] == 0
			bins[i] = 127;
		}
	}

}


//------------------------------------//


__global__ void histoKernel(uint32_t *input, size_t height, size_t width, uint32_t* bins){
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
