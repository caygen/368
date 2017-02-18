#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(int size, uint32_t* dinput, uint8_t* dbins);
void tearDown(void* kernel_bins, void* dbins, void* dinput);
void setUp(void* dinput, void* dbins, int size, uint32_t** input);
__global__ void HistoKernel(int size, uint32_t* dinput, uint8_t* dbins);

#endif
