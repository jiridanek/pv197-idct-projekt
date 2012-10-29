#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "idct_fcpu.h"

/**
* Perform inverse DCT on 8x8 block
*
* @param block
*/
void idct_fcpu_perform(int16_t* block, uint8_t * output)
{
    double Sxy;
    double Cu, Cv = 0.0;
    int x,y,u,v;

    double pi = 3.1415926535897932;

    for (y = 0; y <= 7; y++) {
        for (x = 0; x <= 7; x++) {
            int idx = y*8+x;
            double sxy = 0.0;
            /* coupute idct for position x,y */
            for (u = 0; u <= 7; u++) {
                Cu = (u == 0 ? 1.0/sqrt(2.0) : 1.0);
                for (v = 0; v <= 7; v++) {
                    Sxy = (double)block[v*8+u];
                    Cv = (v == 0 ? 1.0/sqrt(2.0) : 1.0);
                    sxy += Cu * Cv * Sxy * cos (((2.0*x+1.0)*u*pi) / 16.0) * cos (((2.0*y+1.0)*v*pi)/16.0);
                }
            }
            sxy *= 1.0/4.0;

            // <-127:128> -> <0:255>
            sxy += 128.0;

            // Clipping (should not be needed)
            if (sxy < 0.0)   sxy =   0.0; 
            if (sxy > 255.0) sxy = 255.0;

            //Round
            output[idx] = (uint8_t)round(sxy);
        }
	}
}

/** Documented at declaration */
void 
idct_fcpu(int pix_width, int pix_height, int16_t** source, uint8_t** output)
{
    // Perform IDCT and dequantization
    for ( int comp = 0; comp < 3; comp++ ) {
        // Get component
        int16_t* component = source[comp];
        uint8_t* out       = output[comp];
        uint8_t* tmp = (uint8_t*)malloc(pix_width*pix_height);

        // Perform IDCT on CPU
        int width = pix_width / 8;
        int height = pix_height / 8;
        for ( int y = 0; y < height; y++ ) {
            for ( int x = 0; x < width; x++ ) {
                int index = y * width + x;
                idct_fcpu_perform( &component[index * 64], &tmp[index * 64]);
            }
        }

        // Transform lines to visual blocks
        int pos = 0;
        for (int y = 0; y < pix_height; y+=8) {
            for (int x = 0; x < pix_width; x+=8) {
                int idx = y*pix_width + x;

                // Read the block
                for (int i = 0; i < 8; i++) {
                    memcpy(out+idx+i*pix_width, tmp+pos, 8*sizeof(uint8_t));
                    pos += 8;
                }
            }
        }

        free(tmp);
    }
}


