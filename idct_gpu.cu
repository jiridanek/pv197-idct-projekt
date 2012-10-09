#include <stdint.h>
#include <stdio.h>
#include "idct_gpu.h"

/** GPU implementation of inverse 8x8 Discrete Cosine Transform (DCT)
  */ 
__global__ void
idct_gpu_kernel()
{
//   Implmentation of IDCT kernel ...
}

/** Inverse DCT on GPU using CUDA. Function prepares computation, allocates buffers,
  * copy data to GPU, executes GPU kernel and copy data from GPU to 'output' buffer
  *
  * @param pix_width    - Pixel width of input image
  * @param pix_height   - Pixel height of input image
  * @param source       - Pointer to 3 components of input image
  * @param output       - Pointer to 3 output components
  *
  */
void
idct_gpu(int pix_width, int pix_height, int16_t** source, uint8_t** output)
{
    printf("IDCT GPU\n");

    // Alocate GPU buffers 

    // Process three componets
    for (int i = 0; i < 3; i++) {

        int16_t* component; // One component of image. An image is composed of 3
                            // components and each component is processed separately.
        component = source[i];
        
        // Copy data to GPU

        // Prepare kernel configuration like dimension of tread blocks and grid

        // Call DCT Kernel
        /// dct_kernel<<<>>>();

        // Copy data back

    }
}
