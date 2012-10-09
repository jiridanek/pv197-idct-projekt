#ifndef PV197_COMMON_H
#define PV197_COMMON_H


// CUDA check error
#define cuda_check_error(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[GPUJPEG] [Error] %s (line %i): %s: %s.\n", \
                __FILE__, __LINE__, msg, cudaGetErrorString( err) ); \
            exit(-1); \
        } \
    } \

/**
 * Declare timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DECLARE(name) \
    cudaEvent_t name ## _start__; \
    cudaEvent_t name ## _stop__; \
    float name ## _elapsedTime__; \

/**
 * Create timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_CREATE(name) \
    cudaEventCreate(&name ## _start__); \
    cudaEventCreate(&name ## _stop__); \

/**
 * Start timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_START(name) \
    cudaEventRecord(name ## _start__, 0) \

/**
 * Stop timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_STOP(name) \
    cudaEventRecord(name ## _stop__, 0); \
    cudaEventSynchronize(name ## _stop__); \
    cudaEventElapsedTime(&name ## _elapsedTime__, name ## _start__, name ## _stop__) \

/**
 * Get duration for timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DURATION(name) name ## _elapsedTime__

/**
 * Stop timer and print result
 *
 * @param name
 * @param text
 */
#define GPUJPEG_CUSTOM_TIMER_STOP_PRINT(name, text) \
    GPUJPEG_CUSTOM_TIMER_STOP(name); \
    printf("%s %f ms\n", text, name ## _elapsedTime__) \

/**
 * Destroy timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DESTROY(name) \
    cudaEventDestroy(name ## _start__); \
    cudaEventDestroy(name ## _stop__); \

/**
 * Default timer implementation
 */
#define GPUJPEG_TIMER_INIT() \
    GPUJPEG_CUSTOM_TIMER_DECLARE(def) \
    GPUJPEG_CUSTOM_TIMER_CREATE(def)
#define GPUJPEG_TIMER_START() GPUJPEG_CUSTOM_TIMER_START(def)
#define GPUJPEG_TIMER_STOP() GPUJPEG_CUSTOM_TIMER_STOP(def)
#define GPUJPEG_TIMER_DURATION() GPUJPEG_CUSTOM_TIMER_DURATION(def)
#define GPUJPEG_TIMER_STOP_PRINT(text) GPUJPEG_CUSTOM_TIMER_STOP_PRINT(def, text)
#define GPUJPEG_TIMER_DEINIT() GPUJPEG_CUSTOM_TIMER_DESTROY(def)


#endif
