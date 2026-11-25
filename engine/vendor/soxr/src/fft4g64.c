/* SoX Resampler Library      Copyright (c) 2007-13 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

#include <stdlib.h>
#include "filter.h"
#include "fft4g.c"
#include "soxr-config.h"

#if WITH_CR64
static void * null(int length) {(void)length; return 0;}
static void nothing(void * setup) {(void)setup;}
static void forward (int length, void * setup, double * H, void * scratch) {lsx_safe_rdft(length,  1, H); (void)setup; (void)scratch;}
static void backward(int length, void * setup, double * H, void * scratch) {lsx_safe_rdft(length, -1, H); (void)setup; (void)scratch;}
static int multiplier(void) {return 2;}
static void nothing2(int length, void * setup, void * H, void * scratch) {(void)length; (void)setup; (void)H; (void)scratch;}
static int flags(void) {return 0;}

typedef void (* fn_t)(void);
fn_t _soxr_rdft64_cb[] = {
  (fn_t)null,
  (fn_t)null,
  (fn_t)nothing,
  (fn_t)forward,
  (fn_t)forward,
  (fn_t)backward,
  (fn_t)backward,
  (fn_t)_soxr_ordered_convolve,
  (fn_t)_soxr_ordered_partial_convolve,
  (fn_t)multiplier,
  (fn_t)nothing2,
  (fn_t)malloc,
  (fn_t)calloc,
  (fn_t)free,
  (fn_t)flags,
};
#endif
