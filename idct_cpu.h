#ifndef PV197_IDCT_CPU_H
#define PV197_IDCT_CPU_H
#include <stdint.h>

/** Inverse DCT on CPU. Function prepares computation, allocates buffers,
  * copy data to GPU, executes GPU kernel and copy data from GPU to 'output' buffer
  *
  * @param pix_width    - Pixel width of input image
  * @param pix_height   - Pixel height of input image
  * @param source       - Pointer to 3 components of input image
  *
  */
void idct_cpu(int pix_width, int pix_height, int16_t** source);
#endif
