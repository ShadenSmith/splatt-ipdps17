
#include "memory.h"

#ifdef HBW_ALLOC
#include <hbwmalloc.h>
#endif


void * splatt_malloc(
    size_t const bytes)
{
  void * ptr = NULL;
  int ret = posix_memalign(&ptr, SPLATT_MEM_ALIGNMENT, bytes);
  if(ret != 0) {
    fprintf(stderr, "SPLATT: posix_memalign() returned %d.\n", ret);
    assert(0);
  }
  if(ptr == NULL) {
    fprintf(stderr, "SPLATT: posix_memalign() returned NULL.\n");
    assert(0);
  }
  return ptr;
}


void splatt_free(
    void * ptr)
{
  free(ptr);
}


void * splatt_hbw_malloc(
    size_t const bytes)
{
#ifdef HBW_ALLOC
  void * ptr = NULL;
  int ret = hbw_posix_memalign(&ptr, SPLATT_MEM_ALIGNMENT, bytes);

  if(ret != 0) {
    fprintf(stderr, "SPLATT: hbw_posix_memalign() returned %d.\n", ret);
    assert(0);
  }
  if(ptr == NULL) {
    fprintf(stderr, "SPLATT: hbw_posix_memalign() returned NULL.\n");
    assert(0);
  }
  return ptr;

#else
  return splatt_malloc(bytes);
#endif
}


void splatt_hbw_free(
    void * ptr)
{
#ifdef HBW_ALLOC
  hbw_free(ptr);
#else
  splatt_free(ptr);
#endif
}

