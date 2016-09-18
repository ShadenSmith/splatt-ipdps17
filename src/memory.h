#ifndef SPLATT_MEMORY_H
#define SPLATT_MEMORY_H


#include <stdlib.h>


#ifndef SPLATT_MEM_ALIGNMENT
#define SPLATT_MEM_ALIGNMENT 4096
#endif





/*
 * Which components are allocated in HBW?
 */

/* AVX-512 -> use HBW */
#ifdef __AVX512F__
#ifndef SPLATT_HBW_ALLOC
#define SPLATT_HBW_ALLOC 1
#endif

/*
 * Don't use HBW without AXV-512.
 * TODO: Allow toggle HBW witha configure option?
 */
#else
#ifndef SPLATT_HBW_ALLOC
#define SPLATT_HBW_ALLOC 0
#endif

/* if AVX-512 */
#endif

/* define this and run with "numactl -m 0" and MEMKIND_HBW_NODES=1
 * to allocate factor matrices to MCDRAM */
#ifndef SPLATT_MAT_HBW
#define SPLATT_MAT_HBW 1
#endif


/* define this and run with "numactl -m 1" and MEMKIND_HBW_NODES=0
 * to allocate tensor data to DDR */
#ifndef SPLATT_CSF_FPTR_HBW
#define SPLATT_CSF_FPTR_HBW 0
#endif
#ifndef SPLATT_CSF_FIDS_HBW
#define SPLATT_CSF_FIDS_HBW 0
#endif
#ifndef SPLATT_CSF_VALS_HBW
#define SPLATT_CSF_VALS_HBW 0
#endif


/* define this and run with "numactl -m 1" and MEMKIND_HBW_NODES=0
 * to allocate tensor data to DDR */
#ifndef SPLATT_NONPERFORM_HBW
#define SPLATT_NONPERFORM_HBW 0
#endif

/* define this and run with "numactl -m 1" and MEMKIND_HBW_NODES=0
 * to allocate sptensor data to DDR */
#ifndef SPLATT_SPTENSOR_HBW
#define SPLATT_SPTENSOR_HBW 0
#endif


#ifdef __cplusplus
extern "C" {
#endif



/******************************************************************************
 * MEMORY ALLOCATION
 *****************************************************************************/

/**
* @brief Allocate aligned memory. Alignment determined by SPLATT_MEM_ALIGNMENT.
*
* @param bytes The number of bytes to allocate.
*
* @return The allocated memory.
*/
void * splatt_malloc(
    size_t const bytes);


/**
* @brief Free memory allocated by splatt_malloc().
*
* @param ptr The pointer to free.
*/
void splatt_free(
    void * ptr);


/**
* @brief Allocate high-bandwidth memory, if available. Alignment follows
*        SPLATT_MEM_ALIGNMENT.
*
*        If HBW memory is not available, this function defaults to regular
*        splatt_malloc(), returning 64-bit aligned memory.
*
* @param bytes The number of bytes to allocate.
*
* @return The allocated memory.
*/
void * splatt_hbw_malloc(
    size_t const bytes);


/**
* @brief Free memory allocated by splatt_hbw_malloc().
*
* @param ptr The pointer to free.
*/
void splatt_hbw_free(
    void * ptr);


#ifdef __cplusplus
}
#endif


#endif
