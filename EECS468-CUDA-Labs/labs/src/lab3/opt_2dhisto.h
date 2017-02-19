#ifndef OPT_KERNEL
#define OPT_KERNEL


__global__ void HistoKernel(int size, uint32_t* dinput, uint32_t* dbins);

void parallel32to8copy(uint32_t* dbins, uint8_t* dout);

__global__ void CopyKernel(uint32_t* dbins, uint8_t* dout);

#endif
