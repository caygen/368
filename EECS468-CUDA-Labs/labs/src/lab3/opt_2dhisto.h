#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(  d_input, INPUT_HEIGHT, INPUT_WIDTH, d_bins, g_bins );

/* Include below the function headers of any other functions that you implement */
void* AllocateDevice(size_t size);

void CopyToDevice(void* D_device, void* D_host, size_t size);

void CopyFromDevice(void* D_host, void* D_device, size_t size);

void FreeDevice(void* D_device);

#endif
