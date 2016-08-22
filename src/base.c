
#include "base.h"


void * splatt_malloc(
    size_t const bytes)
{
  void * ptr;
  int ret = posix_memalign(&ptr, 4096, bytes);
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
