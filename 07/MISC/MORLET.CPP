// Copyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.
// Reproduction or translation of this work beyond that permitted in
// section 117 of the 1976 United States Copyright Act without the
// express written permission of the copyright owner is unlawful.
// Requests for further information should be addressed to the
// Permission Department, John Wiley & Sons, Inc.  The purchaser may
// make backup copies for his/her own use only and not for distribution
// or resale.  The publisher assumes no responsibility for errors,
// omissions, or damages, caused by the use of these programs or from
// the use of the information contained herein.

/******************************************************************************/
/*                                                                            */
/*  MORLET - Morlet wavelet                                                   */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>

/*
--------------------------------------------------------------------------------

   Morlet - Construct and compute Morlet wavelet

   The constructor computes and saves all wavelet coefficients.
   It also returns several parameters that the user may find useful.

   The function "transform" actually performs the transform.  It may be
   called as many times as desired after the constructor has been called.


   >>> The following parameters are in the constructor parameter list:

   shape - The shape of the mother wavelet is expressed as the ratio of its
     center frequency to the radius of its frequency-domain window
     (which is 0.5 / (sqrt(2) * pi), or about 0.1125).
     This must never be less than 6 or so, as normalization suffers.
     A typical value, used by Morlet himself, is 7.54.  About 4 voices are
     recommended for that value.
     Values much larger than that can require many voices for thorough coverage.
     
   rate - This is the sample rate expressed as a multiple of the Nyquist limit.
     Thus, this parameter cannot be less than 1.  Setting it equal to 1 enables
     resolution at the smallest scale physically possible, but also generates
     much output data.  Further, the smallest-scale coefficients are slightly
     uncentered, leading to tiny but annoying leakage from lower frequencies.
     Also, the discrete shape of this wavelet does not well approximate the
     continuous mother wavelet.  Values larger than 1 are recommended unless
     the tiniest-scale resolution is needed.  Even using 1.5 helps a lot, and
     a value of 2 is as large as needed to avoid all problems.
     Twice the rate is the period (sample points per cycle) of the center
     of response of the smallest-scale wavelet.
     
   nvoices - The number of voices used includes the basic wavelet at each
     scale, so this parameter must be at least one.  Larger values are
     always needed to have a decent frame.  At least 3 or 4 are necessary for
     small values of the shape parameter, with more needed as the shape
     parameter increases.

   nscales - The number of scales includes the mother wavelet, which is the
     smallest scale.  Thus, this parameter must be at least 1.  When actual
     data is transformed, all of these scales may not be computed if the
     data array is too short.  But this sets the upper limit for how many
     can be done.

   border - This is returned as the number of points that will
     be skipped at the beginning and end of the input series for the
     smallest-scale wavelet.  Larger scales will skip more.
     The smallest-scale h(t) vector that is dotted with the input has
     length = 2 * border + 1, so border is its half-length.
     So that the entire vector is used, the transform will start at x[border]
     and will go no further than x[nx-border-1], centering the filter at
     points separated by the distance specified when the transform is called.

   samprate - This is returned as the sample rate: the number of sample
     points per unit time in the basic wavelet equation.  The primary reason
     for returning this value is that rounding it to the nearest integer
     gives a good value for the time-domain spacing parameter for the
     transform routine.

   voicefac - The period of each successive voice at a given scale is
     multipled by this returned quantity.  It is the nvoices'th root of two.

   ok - This is returned 1 if all went well, and 0 if there was insufficient
     memory for storing the FIR filter coefficients or if the user specified
     nonsense parameters.


   >>> The following parameters are in the "transform" parameter list:

   nx - This is the length of the time series input vector x.

   x - Input vector

   spacing - This many sample points will separate each of the smallest-scale
     wavelets in the time domain.  A good value will be near the sample
     rate returned by the constructor.  Smaller values will generate unneeded
     quantities of data.  Larger values cause deterioration of frame quality.
     Values greater than about 1.75 times the sample rate result in serious
     information gaps in the time domain.

   starts - This returned vector is 'nscales' long.  For each scale (0 being
     the smallest), this specifies the subscript in the x array where the first
     wavelet of that scale is centered.  Starts[0] will equal 'border', that
     filter's half-length, as the first wavelet starts as soon as possible.
     The filter half-length doubles for each successive scale, so the
     starting point must move in at least that much.  It may be even more,
     as each longer scale starts at a starting point of the previous (smaller)
     scale to assure uniformity.

   counts - This returned vector is 'nscales' long.  For each scale (0 being
     the smallest), this specifies the number of wavelets computed at
     that scale.

   nout - This is returned as the total of all of the elements of counts.
     Thus, the total number of complex outputs is nout times nvoices.

   rt - Real part of output transform.  It contains nout * nvoices
     elements.

   it - Imaginary part of transform as above.


   The constructor will need to allocate a work area to hold the FIR filter
   coefficients.  This will be 2 * nvoicess * border * 2**(nscales-1) doubles.
   (The filter is complex, symmetric, and the center coefficient is the
   same for all frequencies, so is not stored here.)

--------------------------------------------------------------------------------
*/

static double two_pi = 2. * 3.141592653589793 ;
static double root2 = sqrt ( 2.0 ) ;

class Morlet {

public:
   Morlet ( double shape , double rate , int nvoices , int nscales ,
            int *border , double *samprate , double *voicefac , int *ok ) ;
   ~Morlet () ;
   void transform ( int nx , double *x , int spacing , int *starts ,
                    int *counts , int *nout , double *rt , double *it ) ;

private:
   double kparam ;  // "Frequency" shape parameter k
   double srate ;   // Sample rate (samples per unit time)
   int nv ;         // Number of voices
   int ns ;         // Number of scales
   int hl ;         // Shortest filter half length (=border)
   int npv ;        // Number of coefs per voice (for longest filter)
   double gconst ;  // Normalizing constant for Gaussian
   double *coefs ;  // FIR coefficients for all frequencies
} ;

/*
--------------------------------------------------------------------------------

   Constructor, destructor

--------------------------------------------------------------------------------
*/

Morlet::Morlet (
   double shape ,     // f/delta, Morlet used 7.54
   double rate ,      // Multiple of Nyquist, at least 1.0
   int nvoices ,      // Number of voices, at least 1
   int nscales ,      // Number of scales to prepare, at least 1
   int *border ,      // Half-length of shortest filter
   double *samprate , // Samples per unit time
   double *voicefac , // Factor by which period is multiplied for voices
   int *ok            // Parameters and memory allocation ok?
   )
{
   int i, iv ;
   double window_radius, weight, *cptr, fac, x, con, vfac, rfac ;

   coefs = NULL ;  // So destructor doesn't do bad free if failure here

/*
   Verify that parameters are legal.
   If the shape-determining frequency is very low, normalization is poor
   unless the sample rate is well above the Nyquist limit.
   (This is a crude check, but it inspires more care than no check at all.)
   The specified sample rate must be at least 1.0 times that limit.
   The number of voices and scales must be at least 1.
*/

   *ok = 0 ;
   if ((shape < 6.2)  &&  (rate < 2.0))
      return ;
   if (rate < 1.0)
      return ;
   if (nvoices < 1)
      return ;
   if (nscales < 1)
      return ;

/*
   The user specified the shape as a multiple of the frequency-dimension
   window radius.  Use that to find k.  Multiply k by 2 pi right now
   to avoid having to do it later for trig functions.
   The sample rate is the Nyquist frequency (twice the shape frequency)
   times the user's multiple (which must be at least 1.0).
*/

   window_radius = 1.0 / (root2 * two_pi) ;
   kparam = window_radius * shape * two_pi ;
   *samprate = srate = window_radius * shape * 2.0 * rate ;
   *voicefac = pow ( 2.0 , 1.0 / (double) nvoices ) ;

/*
   Save other user parameters in private storage area.
   Compute the filter half-length such that the weight goes to about
   1.e-12 times its max by the end of the filter.
   If multiple voices are used, remember that we must accomodate the
   largest voice.
*/

   nv = nvoices ;
   ns = nscales ;

   *border = hl = 1 + 7.4 * srate * pow ( 2.0 , (double) (nv-1) / (double) nv );

/*
   Allocate memory for the FIR filter coefficients that make up the
   wavelet family.  We store each voice separately.
   For each voice, store the coefficients for the largest scale member of
   that family.  Smaller scale members will be derived by decimation.
   The number of coefficients needed for that longest filter is the
   half-length of the shortest filter times the largest scale (2 ** (ns-1)).
   Then we have a set for each voice, and real and imaginary parts.
*/

   npv = hl ;     // Length of shortest filter
   i = ns ;       // Number of scales
   while (--i)    // Compute 2 ** (ns-1)
      npv *= 2 ;  // to get length of longest filter

   coefs = (double *) malloc ( 2 * nv * npv * sizeof(double) ) ;
   if (coefs == NULL)
      return ;   // We already initialized ok to 0
   *ok = 1 ;

/*
   Compute the filter coefficients.  The center coefficient for all real
   parts is the voice factor times gconst, the Gaussian multiplier.
   The center of the imaginary part is always 0 (sin 0).
   The sample rate for a voice is the smallest-scale sample rate, srate,
   times the scale of the largest-scale member (since we compute and save
   only its coefs) times the rate factor for that voice.
   The unit time change per point is the reciprocal of that sample rate.
*/

   fac = 1.0 / (srate * pow ( 2.0 , (double) (ns-1))) ;// Largest-scale rate
   rfac = sqrt ( fac ) ;
   gconst = pow ( 3.141592653589793 , -0.25 ) ;  // Pi to the -1/4 power
   gconst *= rfac ;                              // Scaling
   con = exp ( -0.5 * kparam * kparam ) ;        // Centering constant

   for (iv=0 ; iv<nv ; iv++) {     // For all voices
      vfac = pow ( 2.0 , -(double) iv / (double) nv ) ; // Voice factor
      cptr = coefs + iv * 2 * npv ;    // Point to this voice's coef area
      for (i=1 ; i<=npv ; i++) {       // Right half of filter
         x = vfac * fac * i ;
         weight = vfac * gconst * exp ( -0.5 * x * x ) ; // Data window
         *cptr++ = (cos ( kparam * x ) - con) * weight ; // Real part of filter
         *cptr++ = sin ( kparam * x ) * weight ;         // And imaginary part
         }
      }
}

Morlet::~Morlet ()
{
   if (coefs != NULL)
      free ( coefs ) ;
}

void Morlet::transform (
   int nx ,      // Length of input vector
   double *x ,   // Input vector
   int spacing , // Space between smallest-scale time lattice points
   int *starts , // Starts[i] is x subscript of center of first scale i output
   int *counts , // Counts[i] is number of outputs at scale i
   int *nout ,   // Total number of outputs (sum of counts)
   double *rt ,  // Real outputs
   double *it    // Imaginary outputs
   )
{
   int i, n, iscale, iv, itime, decim, flen ;
   double *cptr, *xptr, rsum, isum, vfac, scafac ;

   for (iscale=0 ; iscale<ns ; iscale++)    // Init to all 0 in case x is
      starts[iscale] = counts[iscale] = 0 ; // too short to do all scales
   *nout = 0 ;

   decim = npv / hl ;   // Decimate longest filter to get current filter
   flen = hl ;          // Half length of current filter
   scafac = pow ( 2.0 , 0.5 * (double) (ns-1) ) ; // Scale factor for h(t)

   for (iscale=0 ; iscale<ns ; iscale++) { // Start with shortest filter

      if (iscale) {  // This start must line up with previous and be >= flen
         for (n=starts[iscale-1] ; n<flen ; n+=spacing/2) ;
         starts[iscale] = n ;
         }
      else           // First scale's start is as early as possible
         starts[0] = flen ;

      n = nx - starts[iscale] - flen ;  // This many x's filterable
      if (n > 0) {
         n = counts[iscale] = 1 + (n - 1) / spacing ; // This many resolved
         *nout += n ;
         }
      else
         break ;  // Input is too short given filter length
      
      for (itime=0 ; itime<n ; itime++) {     // All time slots
         xptr = x + starts[iscale] + itime * spacing ; // Center of this filter position

         for (iv=0 ; iv<nv ; iv++) { // All voices
            vfac = pow ( 2.0 , -(double) iv / (double) nv ) ; // Voice factor
            rsum = vfac * gconst * *xptr ;    // Center real coef is constant
            isum = 0.0 ;                      // Center imaginary coef is 0

            cptr = coefs + iv * 2 * npv - 2 ; // Point to this voice coef area
            for (i=1 ; i<=flen ; i++) {       // Right half of filter
               cptr += 2 * decim ;            // Coef 0 not saved in coefs!
               rsum += *cptr * xptr[i] ;      // Sum real part of right half
               isum += *(cptr+1) * xptr[i] ;  // And imaginary part
               }

            cptr = coefs + iv * 2 * npv - 2 ; // Point to this voice coef area
            for (i=1 ; i<=flen ; i++) {       // Left half of filter
               cptr += 2 * decim ;            // Coef 0 not saved in coefs!
               rsum += *cptr * xptr[-i] ;     // Sum real part of left half
               isum -= *(cptr+1) * xptr[-i] ; // And imaginary part
               }

            *rt++ = scafac * rsum ;   // Output real part of Morlet transform
            *it++ = scafac * isum ;   // And imaginary part
            }
         }

      decim /= 2 ;             // Decimation for stored filter coefs
      flen *= 2 ;              // Half-length of current filter
      spacing *= 2 ;           // Time-domain spacing of filters
      scafac *= sqrt ( 0.5 ) ; // Scale factor for h(t)
      }
}

