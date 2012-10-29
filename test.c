#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>


static const int gpujpeg_order_natural[64] = { 
     0,  1,  8, 16,  9,  2,  3, 10, 
    17, 24, 32, 25, 18, 11,  4,  5,  
    12, 19, 26, 33, 40, 48, 41, 34, 
    27, 20, 13,  6,  7, 14, 21, 28, 
    35, 42, 49, 56, 57, 50, 43, 36, 
    29, 22, 15, 23, 30, 37, 44, 51, 
    58, 59, 52, 45, 38, 31, 39, 46, 
    53, 60, 61, 54, 47, 55, 62, 63
};

int main()
{
    int size = 640*480;
    char * filename = "data/output.0-640x480.dct";

    int16_t * buffer  = (int16_t*)malloc(640*480*2);
    int16_t * buffer2 = (int16_t*)malloc(640*480*2);

    printf("Loading %s\n", filename);
    FILE* file;
    file = fopen(filename, "rb");
    if ( !file ) {
        fprintf(stderr, "[Error] Failed open %s for reading!\n", filename);
        exit(-1);
    }

    if ( size != fread(buffer, sizeof(int16_t), size, file) ) {
        fprintf(stderr, "[Error] Failed to load image data [%d bytes] from file %s!\n", size, filename);
        exit(-1);
    }
    fclose(file);

    for (int i = 0; i < 640*480; i+=64) {
        buffer += 64;
        buffer2 += 64;

        for (int j = 0; j < 64; j++) {
            buffer2[j] = buffer[gpujpeg_order_natural[j]];
        }
        
    }

    char filename2[30];
    int k = 0;
    sprintf(filename2, "/tmp/test%d", k);
    FILE* file2;
    file2 = fopen(filename2, "wb");
    if ( !file2 ) {
        fprintf(stderr, "[Error] Failed open %s for reading!\n", filename2);
        exit(-1);
    }

    if ( size != fwrite(buffer2, sizeof(int16_t), size, file2) ) {
        fprintf(stderr, "[Error] Failed to load image data [%d bytes] from file %s!\n", size, filename2);
        exit(-1);
    }
    fclose(file2);

    return 0;
}
