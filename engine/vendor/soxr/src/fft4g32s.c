/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

#include "filter.h"
#include "util32s.h"
#include "rdft_t.h"

static void * null(int length) {(void)length; return 0;}
static void nothing(void * setup) {(void)setup;}
static void forward (int length, void * setup, float * H, void * scratch) {lsx_safe_rdft_f(length,  1, H); (void)setup; (void)scratch;}
static void backward(int length, void * setup, float * H, void * scratch) {lsx_safe_rdft_f(length, -1, H); (void)setup; (void)scratch;}
static int multiplier(void) {return 2;}
static void nothing2(int length, void * setup, void * H, void * scratch) {(void)length; (void)setup; (void)H; (void)scratch;}
static int flags(void) {return RDFT_IS_SIMD;}

fn_t _soxr_rdft32s_cb[] = {
  (fn_t)null,
  (fn_t)null,
  (fn_t)nothing,
  (fn_t)forward,
  (fn_t)forward,
  (fn_t)backward,
  (fn_t)backward,
  (fn_t)ORDERED_CONVOLVE_SIMD,
  (fn_t)ORDERED_PARTIAL_CONVOLVE_SIMD,
  (fn_t)multiplier,
  (fn_t)nothing2,
  (fn_t)SIMD_ALIGNED_MALLOC,
  (fn_t)SIMD_ALIGNED_CALLOC,
  (fn_t)SIMD_ALIGNED_FREE,
  (fn_t)flags,
};
