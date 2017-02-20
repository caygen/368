#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* input, size_t height, size_t width, uint8_t* bins, uint32_t* g_bins);
/* Include below the function headers of any other functions that you implement */
void* AllocateOnDevice(size_t size);
void CopyToDevice(void* d_device, void* d_host, size_t size);
void CopyFromDevice(void* d_host, void* d_device, size_t size);
void FreeCuda(void* d_space);

__global__ void histogram(unsigned int *input, unsigned int *bins,unsigned int num_elements,unsigned int num_bins);
__global__ void saturate(unsigned int *bins, unsigned int num_bins);
#endif
