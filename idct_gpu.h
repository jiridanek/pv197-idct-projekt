#ifndef PV197_IDCT_GPU_H
#define PV197_IDCT_GPU_H
#include <stdint.h>

/** Inverse DCT on GPU using CUDA. Function prepares computation, allocates buffers,
  * copy data to GPU, executes GPU kernel and copy data from GPU to 'output' buffer
  *
  * @param pix_width    - Pixel width of input image
  * @param pix_height   - Pixel height of input image
  * @param source       - Pointer to 3 components of input image
  * @param output       - Pointer to 3 output components
  *
  */
void idct_gpu(int pix_width, int pix_height, int16_t** source, uint8_t** output);
#endif
