#ifndef SIMD_UTILS_H
#define SIMD_UTILS_H

#ifdef __AVX512F__

#define SPLATT_VLEN (8)
#define SPLATT_SIMDFPTYPE __m512d
  
#define _MM_SET1(a) _mm512_set1_pd(a)
#define _MM_SETZERO() _mm512_setzero_pd()

#define _MM_ADD(a, b) _mm512_add_pd(a, b)
#define _MM_MUL(a, b) _mm512_mul_pd(a, b)
#define _MM_FMADD(a, b, c) _mm512_fmadd_pd(a, b, c)
#define _MM_MAX(a, b) _mm512_max_pd(a, b)

#define _MM_LOAD(a) _mm512_load_pd(a)
#define _MM_STORE(a, b) _mm512_store_pd(a, b)
#define _MM_STREAM(a, b) _mm512_stream_pd(a, b)

#else
  /* !__AVX512F__ */

#define SPLATT_VLEN (4)
#define SPLATT_SIMDFPTYPE __m256d

#define _MM_SET1(a) _mm256_set1_pd(a)
#define _MM_SETZERO() _mm256_setzero_pd()

#define _MM_ADD(a, b) _mm256_add_pd(a, b)
#define _MM_MUL(a, b) _mm256_mul_pd(a, b)
#define _MM_FMADD(a, b, c) _mm256_fmadd_pd(a, b, c)
#define _MM_MAX(a, b) _mm256_max_pd(a, b)

#define _MM_LOAD(a) _mm256_load_pd(a)
#define _MM_STORE(a, b) _mm256_store_pd(a, b)
#define _MM_STREAM(a, b) _mm256_stream_pd(a, b)

#endif

#endif
