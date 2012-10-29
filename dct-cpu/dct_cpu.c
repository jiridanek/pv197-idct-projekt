#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

// Load data
void
load_data_from_file(const char* filename, uint8_t* buffer, int size)
{
    FILE* file;
    file = fopen(filename, "rb");
    if ( !file ) {
        fprintf(stderr, "[Error] Failed open %s for reading!\n", filename);
        exit(-1);
    }

    if ( size != fread(buffer, sizeof(uint8_t), size, file) ) {
        fprintf(stderr, "[Error] Failed to load image data [%d bytes] from file %s!\n", size, filename);
        exit(-1);
    }
    fclose(file);
}

void
store_int16_to_int16_file(const char* filename, int16_t* buffer, int size)
{
    // store
    FILE* file;
    file = fopen(filename, "wb");
    if ( !file ) {
        fprintf(stderr, "[Error] Failed open %s for reading!\n", filename);
        exit(-1);
    }

    if ( size != fwrite(buffer, sizeof(int16_t), size, file) ) {
        fprintf(stderr, "[Error] Failed to load image data [%d bytes] from file %s!\n", 2*size, filename);
        exit(-1);
    }
    fclose(file);
}

// Ulozit 16bit zdroj jako 8bit cisla do vystupniho souboru
void
store_int16_to_char_file(const char* filename, int16_t* buffer, int size)
{
    //convert to char
    uint8_t * output = (uint8_t *)malloc(size*sizeof(uint8_t));
    for (int i = 0; i < size; i++) {
        output[i] = buffer[i];
    }

    // store
    FILE* file;
    file = fopen(filename, "wb");
    if ( !file ) {
        fprintf(stderr, "[Error] Failed open %s for reading!\n", filename);
        exit(-1);
    }

    if ( size != fwrite(output, sizeof(uint8_t), size, file) ) {
        fprintf(stderr, "[Error] Failed to load image data [%d bytes] from file %s!\n", size, filename);
        exit(-1);
    }
    fclose(file);
}

/**
 * block - input block 8x8
 * outpu - output block 8x8
 *
 * Pozn.: delam doprednout i zpetnou transformaci. Vysledek dopredne je ulozen
 * v output. Vysledek zpetne se uklada do block. Block by tudiz mel na konci 
 * obsahovat v podstate totozna data.
 */
void dct_fcpu_perform(int16_t* block, int16_t* output)
{
    double Sxy = 0;
    double Cu, Cv = 0;
    int x,y,u,v;

    double pi = 3.1415926535897932;

    //Forward DCT
	for (v = 0; v <= 7; v++) {
		Cv = (v == 0 ? 1.0/sqrt(2.0) : 1.0);
		for (u = 0; u <= 7; u++) {
			Cu = (u == 0 ? 1.0/sqrt(2.0) : 1.0);
			/* computing S_vu */
            double Svu = 0;
			for (x = 0; x <= 7; x++) {
				for (y = 0; y <= 7; y++) {
                    if (block[y*8+x] < -128) printf("input: %d\n", block[y*8+x]);
                    if (block[y*8+x] > 128)  printf("input: %d\n", block[y*8+x]);
					Svu += (double)block[y*8+x] * cos(((2.0*x+1.0)*u*pi)/16.0) * cos(((2.0*y+1.0)*v*pi)/16.0);
				}
			}
			Svu *= 1.0/4.0*Cu*Cv;
            output[v*8 + u] = (int16_t)round(Svu);
		}
	}

    //Reverse DCT
    for (y = 0; y <= 7; y++) {
        for (x = 0; x <= 7; x++) {
            int idx = y*8+x;
            double sxy = 0;
            /* coupute idct for position x,y */
            for (u = 0; u <= 7; u++) {
                Cu = (u == 0 ? 1.0/sqrt(2.0) : 1.0);
                for (v = 0; v <= 7; v++) {
                    Cv = (v == 0 ? 1.0/sqrt(2.0) : 1.0);
                    sxy += Cu * Cv * (double)output[v*8+u] * cos (((2.0*x+1.0)*u*pi) / 16.0) * cos (((2.0*y+1.0)*v*pi)/16.0);
                }
            }
            sxy *= 1.0/4.0;
            block[idx] = (int16_t)round(sxy)+128;
        }
    }
}


int main()
{
    int width  = 640;
    int height = 480;
    int size = width * height;

    uint8_t* input = (char*)malloc(size*sizeof(char));

    // Load data
    char * filename = "../data/big_building_640x480.b";
    load_data_from_file(filename, input, size);

    // Convert to short and normalize around zero
    int16_t * source = (int16_t *)malloc(size*sizeof(int16_t));
    for (int i = 0; i < size; i++) {
        source[i] = (int16_t)input[i]-128;
    }

    // Transform visual blocks to lines
    int16_t * vizblocks = (int16_t *)malloc(size*sizeof(int16_t));
    int pos = 0;
    for (int y = 0; y < height; y+=8) {
        for (int x = 0; x < width; x+=8) {
            int idx = y*width + x;

            // Read the block
            for (int i = 0; i < 8; i++) {
                memcpy(vizblocks+pos, source+idx+i*width, 8*sizeof(int16_t));
                pos += 8;
            }
        }
    }

    // Store transformed file
    store_int16_to_char_file("/tmp/output1.gray", vizblocks, size);

    //DCT
    int16_t * output = (int16_t *)malloc(size*sizeof(int16_t));
    int w = width / 8;
    int h = height / 8;
    for ( int y = 0; y < h; y++ ) {
        for ( int x = 0; x < w; x++ ) {
            int index = y * w + x;
            dct_fcpu_perform( &vizblocks[index * 64], &output[index * 64]);
        }
    }

    // Store DCT file
    store_int16_to_int16_file("../data/big_building_640x480.b.dct", output, size);
    
    // Transform lines to visual blocks
    pos = 0;
    for (int y = 0; y < height; y+=8) {
        for (int x = 0; x < width; x+=8) {
            int idx = y*width + x;

            // Read the block
            for (int i = 0; i < 8; i++) {
                memcpy(output+idx+i*width, vizblocks+pos, 8*sizeof(int16_t));
                pos += 8;
            }
        }
    }

    //Store original file
    store_int16_to_char_file("/tmp/output3.gray", output, size);
}
