
#include "memory.h"

#include <stdio.h>
#include <assert.h>
#include <errno.h>


#if SPLATT_HBW_ALLOC
#include <hbwmalloc.h>
#endif


void * splatt_malloc(
    size_t const bytes)
{
  void * ptr = NULL;
  int ret = posix_memalign(&ptr, SPLATT_MEM_ALIGNMENT, bytes);

  switch(ret) {
  case 0:
    break;
  case ENOMEM:
    fprintf(stderr, "SPLATT: posix_memalign() returned ENOMEM. ");
    fprintf(stderr,"Insufficient memory.");
    assert(0);
    break;
  case EINVAL:
    fprintf(stderr, "SPLATT: posix_memalign() returned EINVAL. ");
    fprintf(stderr,"Alignment %d is invalid.", SPLATT_MEM_ALIGNMENT);
    assert(0);
    break;
  default:
    fprintf(stderr, "SPLATT: posix_memalign() unknown error: %d.", 
        ret);
    assert(0);
    break;
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
#if SPLATT_HBW_ALLOC
  void * ptr = NULL;
  int ret = hbw_posix_memalign(&ptr, SPLATT_MEM_ALIGNMENT, bytes);

  switch(ret) {
  case 0:
    break;
  case ENOMEM:
    fprintf(stderr, "SPLATT: hbw_posix_memalign() returned ENOMEM. ");
    fprintf(stderr,"Insufficient memory.");
    assert(0);
    break;
  case EINVAL:
    fprintf(stderr, "SPLATT: hbw_posix_memalign() returned EINVAL. ");
    fprintf(stderr,"Alignment %d is invalid.", SPLATT_MEM_ALIGNMENT);
    assert(0);
    break;
  default:
    fprintf(stderr, "SPLATT: hbw_posix_memalign() unknown error: %d.", 
        ret);
    assert(0);
    break;
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
#if SPLATT_HBW_ALLOC
  hbw_free(ptr);
#else
  splatt_free(ptr);
#endif
}

