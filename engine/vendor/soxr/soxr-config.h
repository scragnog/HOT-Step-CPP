/* SoX Resampler Library      Copyright (c) 2007-16 robs@users.sourceforge.net
 * Licence for this file: LGPL v2.1                  See LICENCE for details. */

/* HOT-Step: soxr-config.h.in filled in by hand with python-soxr 1.0.0's build options
 * (libsoxr 0.1.3-14-ga66f3ee: no OpenMP, no LSR bindings, no VR32, PFFFT, no libavcodec),
 * minus the 64-bit engines: HQ (20-bit) never reaches them. */

#if !defined soxr_config_included
#define soxr_config_included

#define AVCODEC_FOUND 0
#define AVUTIL_FOUND 0
#define WITH_PFFFT 1

#define HAVE_FENV_H 1
#define HAVE_STDBOOL_H 1
#define HAVE_STDINT_H 1
#if defined __linux__
#define HAVE_GETAUXVAL 1
#else
#define HAVE_GETAUXVAL 0
#endif
#define HAVE_LRINT 1
#define HAVE_BIGENDIAN 0

#define WITH_CR32 1
#define WITH_CR32S 1
#define WITH_CR64 0
#define WITH_CR64S 0
#define WITH_VR32 0

#define WITH_HI_PREC_CLOCK 1
#define WITH_FLOAT_STD_PREC_CLOCK 0
#define WITH_DEV_TRACE 1

#endif
