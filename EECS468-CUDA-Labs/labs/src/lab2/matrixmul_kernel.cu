/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_
#define BLOCK_SIZE 16 
#define TILE_DIM 16

#include <stdio.h>
#include "matrixmul.h"
//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"
////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) { 
	Matrix Asub; 
	Asub.width = BLOCK_SIZE;
	Asub.height = BLOCK_SIZE; 
	Asub.pitch = A.pitch; 
	Asub.elements = &A.elements[A.pitch * BLOCK_SIZE * row + BLOCK_SIZE * col]; 
	return Asub;
}
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)

{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
 //   for (int i = 0; i < TILE_DIM ; i++){
//	    cuprintf("%f", As[i]); 
//	}
    for (int k = 0; k < (TILE_DIM + M.width - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < M.width && Row < M.height)
             As[threadIdx.y][threadIdx.x] = M.elements[Row*M.width + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < N.height && Col < N.width)
             Bs[threadIdx.y][threadIdx.x] = N.elements[(k*TILE_DIM + threadIdx.y)*N.width + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < P.height && Col < P.width)
        P.elements[((blockIdx.y * blockDim.y + threadIdx.y)*P.width) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
/*
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;	
	int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

	__shared__ float pblock[TILE_WIDTH][TILE_WIDTH];
	__shared__ float mblock[TILE_WIDTH][TILE_WIDTH];
	__shared__ float nblock[TILE_WIDTH][TILE_WIDTH];
*/
/*	// Block row and column
	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	// Each thread block computes one sub-matrix Csub of C
	Matrix Csub = GetSubMatrix(P, blockRow, blockCol);

	// Each thread computes one element of Csub
	// by accumulating results into Cvalue
	float Cvalue = 0;
// Thread row and column within Csub int row = threadIdx.y;
	int col = threadIdx.x;

	// Loop over all the sub-matrices of A and B that are
	// required to compute Csub
	// Multiply each pair of sub-matrices together
	// and accumulate the results
	for (int m = 0; m < (M.width / BLOCK_SIZE); ++m) {

		// Get sub-matrix Asub of A
		Matrix Asub = GetSubMatrix(M, blockRow, m);

		// Get sub-matrix Bsub of B
		Matrix Bsub = GetSubMatrix(N, m, blockCol);

		// Shared memory used to store Asub and Bsub respectively
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		// Load Asub and Bsub from device memory to shared memory
		// Each thread loads one element of each sub-matrix
		As[row][col] = Asub.elements[row * Asub.pitch + col]; 
		Bs[row][col] = Bsub.elements[row * Bsub.pitch + col];

		// Synchronize to make sure the sub-matrices are loaded
		// before starting the computation
		__syncthreads();

		// Multiply Asub and Bsub together
		for (int e = 0; e < BLOCK_SIZE; ++e){
			Cvalue += As[row][e] * Bs[e][col];
		}
	
		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write Csub to device memory
	// Each thread writes one element
	Csub.elements[row * Csub.pitch + col] = Cvalue;
*/
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
