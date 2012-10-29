#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "idct_cpu.h"

#define W1 2841 // 2048*sqrt(2)*cos(1*pi/16)
#define W2 2676 // 2048*sqrt(2)*cos(2*pi/16)
#define W3 2408 // 2048*sqrt(2)*cos(3*pi/16)
#define W5 1609 // 2048*sqrt(2)*cos(5*pi/16)
#define W6 1108 // 2048*sqrt(2)*cos(6*pi/16)
#define W7 565  // 2048*sqrt(2)*cos(7*pi/16)

/** Clipping table and pointer to it */
static int16_t* iclip;//[1024];
static int16_t* iclp;

/**
 * Row (horizontal) IDCT
 *
 *           7                       pi         1
 * dst[k] = sum c[l] * src[l] * cos( -- * ( k + - ) * l )
 *          l=0                      8          2
 *
 * where: c[0]    = 128
 *        c[1..7] = 128*sqrt(2)
 */
void
idct_cpu_perform_row(int16_t* blk)
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;

    // shortcut
    if (!((x1 = blk[4]<<11) | (x2 = blk[6]) | (x3 = blk[2]) |
        (x4 = blk[1]) | (x5 = blk[7]) | (x6 = blk[5]) | (x7 = blk[3])))
    {
        blk[0]=blk[1]=blk[2]=blk[3]=blk[4]=blk[5]=blk[6]=blk[7]=blk[0]<<3;
        return;
    }

    // for proper rounding in the fourth stage
    x0 = (blk[0]<<11) + 128;

    // first stage
    x8 = W7*(x4+x5);
    x4 = x8 + (W1-W7)*x4;
    x5 = x8 - (W1+W7)*x5;
    x8 = W3*(x6+x7);
    x6 = x8 - (W3-W5)*x6;
    x7 = x8 - (W3+W5)*x7;

    // second stage
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2);
    x2 = x1 - (W2+W6)*x2;
    x3 = x1 + (W2-W6)*x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    // third stage
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181*(x4+x5)+128)>>8;
    x4 = (181*(x4-x5)+128)>>8;

    // fourth stage
    blk[0] = (x7+x1)>>8;
    blk[1] = (x3+x2)>>8;
    blk[2] = (x0+x4)>>8;
    blk[3] = (x8+x6)>>8;
    blk[4] = (x8-x6)>>8;
    blk[5] = (x0-x4)>>8;
    blk[6] = (x3-x2)>>8;
    blk[7] = (x7-x1)>>8;
}

/**
 * Column (vertical) IDCT
 *
 *             7                         pi         1
 * dst[8*k] = sum c[l] * src[8*l] * cos( -- * ( k + - ) * l )
 *            l=0                        8          2
 *
 * where: c[0]    = 1/1024
 *        c[1..7] = (1/1024)*sqrt(2)
 */
void
idct_cpu_perform_column(int16_t* blk)
{
    int x0, x1, x2, x3, x4, x5, x6, x7, x8;

    // shortcut
    if (!((x1 = (blk[8*4]<<8)) | (x2 = blk[8*6]) | (x3 = blk[8*2]) |
        (x4 = blk[8*1]) | (x5 = blk[8*7]) | (x6 = blk[8*5]) | (x7 = blk[8*3])))
    {
        blk[8*0]=blk[8*1]=blk[8*2]=blk[8*3]=blk[8*4]=blk[8*5]=blk[8*6]=blk[8*7]=
                iclp[(blk[8*0]+32)>>6];
        return;
    }

    x0 = (blk[8*0]<<8) + 8192;

    // first stage
    x8 = W7*(x4+x5) + 4;
    x4 = (x8+(W1-W7)*x4)>>3;
    x5 = (x8-(W1+W7)*x5)>>3;
    x8 = W3*(x6+x7) + 4;
    x6 = (x8-(W3-W5)*x6)>>3;
    x7 = (x8-(W3+W5)*x7)>>3;

    // second stage
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6*(x3+x2) + 4;
    x2 = (x1-(W2+W6)*x2)>>3;
    x3 = (x1+(W2-W6)*x3)>>3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;

    // third stage
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181*(x4+x5)+128)>>8;
    x4 = (181*(x4-x5)+128)>>8;

    // fourth stage
    blk[8*0] = iclp[(x7+x1)>>14];
    blk[8*1] = iclp[(x3+x2)>>14];
    blk[8*2] = iclp[(x0+x4)>>14];
    blk[8*3] = iclp[(x8+x6)>>14];
    blk[8*4] = iclp[(x8-x6)>>14];
    blk[8*5] = iclp[(x0-x4)>>14];
    blk[8*6] = iclp[(x3-x2)>>14];
    blk[8*7] = iclp[(x7-x1)>>14];
}

/**
 * Perform inverse DCT on 8x8 block
 *
 * @param block
 */
void idct_cpu_perform(int16_t* block)
{
    for ( int i = 0; i < 8; i++ )
        idct_cpu_perform_row(block + 8 * i);

    for ( int i = 0; i < 8; i++ )
        idct_cpu_perform_column(block + i);
}

/**
 * Init inverse DCT
 */
void idct_cpu_init()
{
    iclip = (int16_t *)malloc(1024*2);
    iclp = iclip + 512;
    for ( int i = -512; i < 512; i++ )
        iclp[i] = (i < -256) ? -256 : ((i > 255) ? 255 : i);
}

/** Documented at declaration */
void 
idct_cpu(int pix_width, int pix_height, int16_t** source, uint8_t** output)
{
    idct_cpu_init();

    // Perform IDCT and dequantization
    for ( int comp = 0; comp < 3; comp++ ) {
        // Get component
        int16_t* component = source[comp];

        // Perform IDCT on CPU
        int width = pix_width / 8;
        int height = pix_height / 8;
        for ( int y = 0; y < height; y++ ) {
            for ( int x = 0; x < width; x++ ) {
                int index = y * width + x;
                idct_cpu_perform( &component[index * 64]);
            }
        }

        uint8_t* out = output[comp];
        uint8_t* tmp = (uint8_t*)malloc(pix_width*pix_height);

        //convert to uint
        for (int i = 0; i < pix_width*pix_height; i++) {
            tmp[i] = (uint8_t)component[i]+128;
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

    free(iclip);
}
