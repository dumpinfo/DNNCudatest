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
/*  GABOR - Gabor transform                                                   */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>

/*
--------------------------------------------------------------------------------

   Gabor - Construct and compute Gabor transform

   The constructor computes and returns sigma, delta_t, and delta_f as
   discussed below.  It also allocates memory for holding the FIR filter
   coefficients that define the frame elements.

   The function "transform" actually performs the transform.  It may be
   called as many times as desired after the constructor has been called.

   The sample interval defines the time unit throughout this routine.


   >>> The following parameters are in the constructor parameter list:

   tau0 - Number of samples separating each lattice point in the time dimension

   phi0 - Frequency increment separating the lattice points in the frequency
     dimension.  Its theoretical upper limit to have a frame is 1/tau0,
     but it really should be half that or less for good coverage.

   sigma, delta_t, delta_f - Scale factor, and window radii.
     If all are input negative, sigma is computed as its optimal value (which
     produces a resolution window whose shape matches the lattice spacing).
     If sigma is input positive (regardless of the deltas), that value is used.
     In either case, the deltas are computed according to sigma.
     If sigma is input negative and delta_t is input positive, that value of
     delta_t is used, and sigma and delta_f are returned accordingly.
     If sigma and delta_t are input negative, and delta_f is input positive,
     that value will be used to compute sigma and delta_t.

   border - This is returned as the number of doubles that will
     be skipped at the beginning and end of the input series.  The g(t)
     vector that is dotted with the input series has length = 2 * border + 1.
     So that the entire vector is used, the transform will start at x[border]
     and will go no further than x[nx-border-1], centering the filter at every
     tau0'th point.

   nfreqs - This is returned as the number of frequency bands that will be
     resolved.  It is 1 + 0.5 / phi0, truncated down to an integer.
     This takes us as high as we can go without passing the Nyquist limit.

   ok - This is returned 1 if all went well, and 0 if there was insufficient
     memory for storing the FIR filter coefficients.


   >>> The following parameters are in the "transform" parameter list:

   nx - This is the length of the time series input vector x.

   x - Input vector

   nt - This is returned as the number of time slots that will be
     resolved.  The transformable extent of x runs from x[border]
     through x[nx-border-1].  The transformation filter will be applied
     first to x[border], then to x[border+tau0], continuing through
     x[border+(nt-1)*tau0].  Thus, nt = 1 + (nx-2*border-1) / tau0.

   rt - Real part of output transform.  It contains ntimes * nfreqs
     elements. The first nfreqs elements correspond to x[border]
     for frequencies from 0 through (nfreqs-1)*phi0.
     The next nfreqs are the same frequencies for x[border+tau0].
     The last nfreqs are the same frequencies for x[border+(nt-1)*tau0].

   it - Imaginary part of transform as above.


   The constructor will need to allocate a work area to hold the FIR filter
   coefficients.  This will be 2 * nfreqs * border doubles.
   (The filter is complex, symmetric, and the center coefficient is the
   same for all frequencies, so is not stored here.)

--------------------------------------------------------------------------------
*/

static double two_pi = 2. * 3.141592653589793 ;
static double root2 = sqrt ( 2.0 ) ;

class Gabor {

public:
   Gabor ( int tau0 , double phi0 , double *sigma , double *delta_t ,
           double *delta_f , int *border , int *nfreqs , int *ok ) ;
   ~Gabor () ;
   void transform ( int nx , double *x , int *nt , double *rt , double *it ) ;

private:
   int t0 ;         // Tau0
   double f0 ;      // Phi0
   double sig ;     // Sigma
   double dt ;      // Delta_t
   double df ;      // Delta_f
   int hl ;         // Filter half length (=border)
   int nf ;         // Number of frequencies resolved (=nfreqs)
   double gconst ;  // Normalizing constant for Gaussian
   double *coefs ;  // FIR coefficients for all frequencies
} ;

/*
--------------------------------------------------------------------------------

   Constructor, destructor

--------------------------------------------------------------------------------
*/

Gabor::Gabor (
   int tau0 ,          // Lattice spacing in time dimension
   double phi0 ,       // Lattice spacing in frequency dimension
   double *sigma ,     // Data window scale factor
   double *delta_t ,   // Radius of time-domain window
   double *delta_f ,   // Radius of frequency-domain window
   int *border ,       // Filter half-length (skip this at ends)
   int *nfreqs ,       // Number of frequencies resolved
   int *ok             // Memory allocation ok?
   )
{
   int ifreq, ipos ;
   double freq, weight, fac, *cptr ;

   t0 = tau0 ;   // Keep a private copy of these parameters
   f0 = phi0 ;

/*
   Compute (as needed) sigma, delta_t, and delta_f
*/

   if (*sigma <= 0.0) {
      if (*delta_t <= 0.0) {
         if (*delta_f <= 0.0) {
            *sigma = sig = sqrt ( (double) tau0 / (phi0 * two_pi) ) ;
            *delta_t = dt = sig / root2 ;
            *delta_f = df = 1.0 / (two_pi * root2 * sig) ;
            }
         else {
            df = *delta_f ;
            *sigma = sig = 1.0 / (two_pi * root2 * df) ;
            *delta_t = dt = sig / root2 ;
            }
         }
      else {
         dt = *delta_t ;
         *sigma = sig = root2 * dt ;
         *delta_f = df = 1.0 / (two_pi * root2 * sig) ;
         }
      }
   else {
      sig = *sigma ;
      *delta_t = dt = sig / root2 ;
      *delta_f = df = 1.0 / (two_pi * root2 * sig) ;
      }

/*
   Compute the filter half-length such that the weight goes to about
   1.e-12 times its max at the end of the filter.
   Compute the number of resolved frequencies such that we go as high
   as possible, without exceeding the Nyquist frequency of 0.5.
*/

   *border = hl = 1 + 7.4 * sig ;  // Goes to about 1.e-12
   *nfreqs = nf = 1 + 0.5 / f0 ;

   coefs = (double *) malloc ( 2 * nf * hl * sizeof(double) ) ;
   if (coefs == NULL) {
      *ok = 0 ;
      return ;
      }
   *ok = 1 ;

/*
   Compute the filter coefficients.  The center coefficient for all real
   parts is gconst, the Gaussian multiplier, regardless of frequency.
   The center of the imaginary part is always 0 (sin 0).
*/

   gconst = exp ( 0.25 * log ( 3.141592653589793 )) ;  // Pi to the 1/4 power
   gconst = 1.0 / (gconst * sqrt ( sig )) ;  // Actual normalizer for Gaussian

   fac = -0.5 / (sig * sig) ;            // Common factor in exponent

   for (ifreq=0 ; ifreq<nf ; ifreq++) {  // Covers 0 to Nyquist
      freq = ifreq * f0 * two_pi ;       // Freq term in complex exponential
      cptr = coefs + ifreq * 2 * hl ;    // Point to this freq's coef area
      for (ipos=1 ; ipos<=hl ; ipos++) { // Right half of filter
         weight = gconst * exp ( fac * ipos * ipos ) ; // Data window
         *cptr++ = cos ( ipos * freq ) * weight ; // Real part of filter
         *cptr++ = sin ( ipos * freq ) * weight ; // And imaginary part
         }
      }
}

Gabor::~Gabor ()
{
   if (coefs != NULL)
      free ( coefs ) ;
}

void Gabor::transform (
   int nx ,      // Length of input vector
   double *x ,   // Input vector
   int *nt ,     // Number of time points resolved
   double *rt ,  // Real output
   double *it    // Imaginary output
   )
{
   int ifreq, itime, ipos ;
   double *cptr, *xptr, rsum, isum ;

   *nt = 1 + (nx - 2 * hl - 1) / t0 ;       // This many times resolved

   for (itime=0 ; itime<*nt ; itime++) {    // All time slots
      xptr = x + hl + itime * t0 ;          // Center of this filter position

      for (ifreq=0 ; ifreq<nf ; ifreq++) {  // Covers 0 to Nyquist
         rsum = gconst * *xptr ;            // Center real coef is gconst
         isum = 0.0 ;                       // Center imaginary coef is 0

         cptr = coefs + ifreq * 2 * hl ;    // Point to this freq's coef area
         for (ipos=1 ; ipos<=hl ; ipos++) { // Right half of filter
            rsum += *cptr++ * xptr[ipos] ;  // Cumulate real part of right half
            isum += *cptr++ * xptr[ipos] ;  // And imaginary part
            }

         cptr = coefs + ifreq * 2 * hl ;    // Point to this freq's coef area
         for (ipos=1 ; ipos<=hl ; ipos++) { // Left half of filter
            rsum += *cptr++ * xptr[-ipos] ; // Cumulate real part of left half
            isum -= *cptr++ * xptr[-ipos] ; // And imaginary part
            }

         *rt++ = rsum ;     // Output real part of Gabor transform
         *it++ = isum ;     // And imaginary part
         }
      }
}

