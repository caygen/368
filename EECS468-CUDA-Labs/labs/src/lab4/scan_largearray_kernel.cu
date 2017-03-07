#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 1024
#define scale 2
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void scan(float *g_odata, float *g_idata, int blockSize);
__global__ void copy(float *from_scanned,float *from_unscanned, float *to, int numCopies);
__global__ void addArray(float *from, float *to, int divisor, int numElements);

__global__ void scan2(float *g_odata, float *g_idata, int n);

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements,
                  float *array1, float *array2, float *array3, float *array4)
{
	// each block computes 1024 indices, so 2 indices per thread
	// do scan on inArray, write back to outArray

	scan<<<16384, 512>>>(outArray, inArray, 1024);
  //scan2<<<16384*scale, 256/scale>>>(outArray, inArray, 1024);

	// read every 1024 elements from outArray and write it to array1
	copy<<<16, 1024>>>(outArray, inArray, array1, 16384);

	// do scan on array1, write back to array2
	scan<<<16, 512>>>(array2, array1, 1024);
  //scan2<<<16*scale, 512/scale>>>(array2, array1, 1024);

	// read every 1024 elements from array2 and write it to array3
	copy<<<1, 16>>>(array2, array1, array3, 16);

	// do scan on array3, write back to array4
	scan<<<1, 8>>>(array4, array3, 16);
	//scan2<<<1*scale, 8/scale>>>(array4, array3, 16);

	// add array2[i] to outArray[i*1048576:((i+1)*1048576)-1]
	addArray<<<16384, 1024>>>(array4, outArray, 1024*1024, numElements);

	// add array1[i] to outArray[i*1024:((i+1)*1024)-1]
	addArray<<<16384, 1024>>>(array2, outArray, 1024, numElements);

}
// **===-----------------------------------------------------------===**


// read every 1024 elements from from, and write it to to
__global__ void copy(float *from_scanned, float *from_unscanned, float *to, int numCopies)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int numThreads = gridDim.x * blockDim.x;
	//if (thid < numCopies)
	for (int i = thid; i < numCopies; i += numThreads)
	{
		//to[i] = from[(i + 1) * 1024 - 1];
		to[i] = from_scanned[(i + 1) * 1024 - 1] + from_unscanned[(i + 1) * 1024 - 1];
	}
}

// do scan on g_idata, write back to g_odata
__global__ void scan(float *g_odata, float *g_idata, int blockSize)
{
	__shared__ float temp[1024];
	int allid = blockIdx.x * blockDim.x + threadIdx.x;
	int thid = threadIdx.x;
	int offset = 1;

	temp[2*thid] = g_idata[2*allid]; // load input into shared memory
	temp[2*thid+1] = g_idata[2*allid+1];

	for (int d = blockSize>>1; d > 0; d >>= 1)    // build sum in place up the tree
	{
		__syncthreads();
   		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid == 0) { temp[blockSize - 1] = 0; } // clear the last element

	for (int d = 1; d < blockSize; d *= 2) // traverse down tree & build scan
	{
     		offset >>= 1;
     		__syncthreads();
     		if (thid < d)
     		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
 	__syncthreads();

	g_odata[2*allid] = temp[2*thid]; // write results to device memory
	g_odata[2*allid+1] = temp[2*thid+1];
}

// add from[i] to to[i*divisor:((i+1)*divisor)-1]
__global__ void addArray(float *from, float *to, int divisor, int numElements)
{
	int numThreads = gridDim.x * blockDim.x;
	int thid = blockDim.x * blockIdx.x + threadIdx.x;
	for (int i = thid; i < numElements; i += numThreads)
	{
		to[i] += from[(i / divisor)];
	}
}

//-====================================================================
//-====================================================================

// below is scan optimized for bank conflicts
__global__ void scan2(float *g_odata, float *g_idata, int n)
{
	__shared__ float temp[1024];
	int thid = threadIdx.x;
  //int thid = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = 1;

	int ai = thid;
	//int bi = thid + (n/2);
  int bi = thid + (blockDim.x /2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];  // load input into shared memory


	for (int d = n>>1; d > 0; d >>= 1)    // build sum in place up the tree
	{
		__syncthreads();
   		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	if (thid==0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;}  // clear the last element

	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
     		offset >>= 1;
     		__syncthreads();
     		if (thid < d)
     		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
 	__syncthreads();

	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];  // write results to device memory
}



#endif // _PRESCAN_CU_
