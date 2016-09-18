#ifndef SPLATT_MEMORY_H
#define SPLATT_MEMORY_H

#ifndef SPLATT_MEM_ALIGNMENT
#define SPLATT_MEM_ALIGNMENT 4096
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
