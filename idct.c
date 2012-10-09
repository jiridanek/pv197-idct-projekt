#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "common.h"
#include "idct_gpu.h"
#include "idct_cpu.h"

// Load data
int
load_data_from_file(const char* filename, int16_t* buffer, int size)
{
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
}

// Compare GPU and CPU outputs
int 
idct_cpu_compare(int width, int height, int16_t** component, uint8_t** output_from_gpu)
{
    // IDCT on CPU
    idct_cpu(width, height, component);
        
    // Compare results
    for (int j; j < 3; j++) {
        int16_t * c = component[j];
        uint8_t * o = output_from_gpu[j];
        for (int i; i < width*height; i++) {
            if (c[i] != o[i]) {
                printf("Results does not match: %d != %d, position %d\n", c[i], o[i], i);
                return -1;
            }
        }
    }
}

void
print_help() 
{
    printf(
        "idct [options] comp0.dct comp1.dct comp2.dct]\n"
        "   -h, --help             print help\n"
        "   -D, --device           set cuda device id (default 0)\n"
        "   -s, --size             set input image size in pixels, e.g. 1920x1080\n"
    );
}

// Initialize GPU device
int
init_gpu_device(int device_id)
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    if ( dev_count == 0 ) {
        fprintf(stderr, "[Error] No CUDA enabled device\n");
        return -1;
    }

    if ( device_id < 0 || device_id >= dev_count ) {
        fprintf(stderr, "[Error] Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
            device_id, 0, dev_count - 1);
        return -1;
    }

    struct cudaDeviceProp devProp;
    if ( cudaSuccess != cudaGetDeviceProperties(&devProp, device_id) ) {
        fprintf(stderr,
            "[Error] Can't get CUDA device properties!\n"
            "[Error] Do you have proper driver for CUDA installed?\n"
        );
        return -1;
    }

    if ( devProp.major < 1 ) {
        fprintf(stderr, "[Error] Device %d does not support CUDA\n", device_id);
        return -1;
    }
    
    int cuda_driver_version = 0;
    cudaDriverGetVersion(&cuda_driver_version);
    printf("CUDA driver version:   %d.%d\n", cuda_driver_version / 1000, (cuda_driver_version % 100) / 10);
    
    int cuda_runtime_version = 0;
    cudaRuntimeGetVersion(&cuda_runtime_version);
    printf("CUDA runtime version:  %d.%d\n", cuda_runtime_version / 1000, (cuda_runtime_version % 100) / 10);
    
    printf("Using Device #%d:       %s (c.c. %d.%d)\n", device_id, devProp.name, devProp.major, devProp.minor);
    
    cudaSetDevice(device_id);
    cuda_check_error("Set CUDA device");

    // Test by simple copying that the device is ready
    uint8_t data[] = {8};
    uint8_t* d_data = NULL;
    cudaMalloc((void**)&d_data, 1);
    cudaMemcpy(d_data, data, 1, cudaMemcpyHostToDevice);
    cudaFree(d_data);
    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ) {
        fprintf(stderr, "[Error] Failed to initialize CUDA device.\n");
        return -1;
    }

    return 0;
}

// Main
int
main(int argc, char *argv[])
{
    // Parameters
    int device_id = 0;
    int width = 0;
    int height = 0;

    struct option longopts[] = {
        {"help",                    no_argument,       0, 'h'},
        {"device",                  required_argument, 0, 'D'},
        {"size",                    required_argument, 0, 's'},
        0
    };

    // Parse command line
    char ch = '\0';
    int optindex = 0;
    char* pos = 0;
    while ( (ch = getopt_long(argc, argv, "hD:s:", longopts, &optindex)) != -1 ) {
        switch (ch) {
        case 'h':
            print_help();
            return 0;
        case 's':
            width = atoi(optarg);
            pos = strstr(optarg, "x");
            if ( pos == NULL || width == 0 || (strlen(pos) >= strlen(optarg)) ) {
                print_help();
                return -1;
            }
            height = atoi(pos + 1);
            break;
        case 'D':
            device_id = atoi(optarg);
            break;
        case '?':
            return -1;
        default:
            print_help();
            return -1;
        }
    }
    argc -= optind;
    argv += optind;

    // 3 source components must be present
    if ( argc < 3 ) {
        fprintf(stderr, "Please supply 3 source components filenames!\n");
        print_help();
        return -1;
    }
    
    // Init device
    if ( init_gpu_device(device_id) != 0 )
        return -1;

    // Buffers
    int size = width*height;
    int16_t* component[3];
    uint8_t* output[3];

    // Allocate component buffers
    for (int i = 0; i < 3; i++) {
        cudaMallocHost((void**)&component[i], width*height * sizeof(int16_t));
        cuda_check_error("Source data buffer allocation");
    }
    
    // Allocate output buffers
    for (int i = 0; i < 3; i++) {
        cudaMallocHost((void**)&output[i], width*height * sizeof(uint8_t));
        cuda_check_error("Output data buffer allocation");
    }

    // Load data
    load_data_from_file(argv[0], component[0], size);
    load_data_from_file(argv[1], component[1], size);
    load_data_from_file(argv[2], component[2], size);
    
    // Timing
    GPUJPEG_TIMER_INIT();

    // Call GPU
    printf("Calling GPU IDCT\n");
    GPUJPEG_TIMER_START();
    idct_gpu(width, height, component, output);
    GPUJPEG_TIMER_STOP();
    
    // Print Timing
    printf("IDCT:          %10.2f ms\n", GPUJPEG_TIMER_DURATION());

    // Call CPU compare
    if ( idct_cpu_compare(width, height, component, output) == 0 ) {
        printf("CPU Test: OK\n");
    } else {
        printf("CPU Test: Output from GPU differs\n");
    }
}




