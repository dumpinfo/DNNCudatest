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
/*  GABOR2D - Two-dimensional Gabor transform                                 */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>

/*
--------------------------------------------------------------------------------

   Gabor2D - Construct and compute 2D Gabor transform

   The constructor computes and returns sigma, delta_t, and delta_f as
   discussed below.  It also allocates memory for holding the FIR filter
   coefficients that define the frame elements.

   The function "transform" actually performs the transform.  It may be
   called as many times as desired after the constructor has been called.

   The sample interval defines the time unit throughout this routine.


   >>> The following parameters are in the constructor parameter list:

   sigma, delta_t, delta_f - Scale factor and window radii.
     If sigma is input positive (regardless of the deltas), that value is used.
     The deltas are then computed according to sigma.
     If sigma is input negative and delta_t is input positive, that value of
     delta_t is used, and sigma and delta_f are returned accordingly.
     If sigma and delta_t are input negative, and delta_f is input positive,
     that value will be used to compute sigma and delta_t.

   nfreqs - This is returned as the number of frequency bands that will be
     resolved, including a frequency of zero.  They will be equally spaced
     up to the Nyquist limit.

   ndir - Number of orientation directions, at least 1.  If more than 1, they
     will be equally spaced around a half circle, with the first aligned
     at 'orient'.

   orient - Orientation angle in degrees (0-180) of the first filter.

   halflen - This is returned as the filter half-length.  It is the number of
     rows and columns in all directions that will be accessed by the transform
     routine.  A very serious runtime error (possibly crashing the system) could
     occur if the transform is applied closer to an edge than this quantity.
     Therefore, the transform routine aborts if this is attempted.

   ok - This is returned 1 if all went well, and 0 if there was insufficient
     memory for storing the FIR filter coefficients.


   >>> The following parameters are in the "transform" parameter list:

   nrows - Number of rows in the image

   ncols - Number of columns in the image

   *img - The image.  It must be stored as a vector with the first (top)
     row first, then the second row, et cetera.

   rcent - Row (origin 0) where the filter is to be centered.  It must be
     greater than or equal to halflen, the filter half length.  It must be
     strictly less than nrows - halflen.  This prevents the filter from
     spilling outside the image.  If a closer approach to the edges is
     needed, pad the image with zeros.

   ccent - Column center as above.

   *rt - Real part of the output transform.  This is a vector nfreqs * ndirs
     long.  All orientation directions for zero frequency are first.  (This
     is a little silly, as they are all the same!)

   *it - And imaginary part as above.


   The constructor will need to allocate a work area to hold the FIR filter
   coefficients.  This is 2 * (halflen+1) * (2*halflen+1) * nfreqs * ndir
   doubles.  Each coefficient is complex (2 doubles).  There will be
   nfreqs * ndir filters saved.  Only the center row and upper half of each
   will be found.  Each row is 2*halflen+1 long, and there are halflen+1 rows.
   An element in the bottom half of the filter is the complex conjugate of the
   element reflected along both axes.

--------------------------------------------------------------------------------
*/

static double pi = 3.141592653589793 ;
static double two_pi = 2. * pi ;
static double root2 = sqrt ( 2.0 ) ;
static double rootpi = sqrt ( pi ) ;

class Gabor2D {

public:
   Gabor2D ( double *sigma , double *delta_t , double *delta_f ,
             int nfreqs , int ndir , double orient , int *halflen , int *ok ) ;
   ~Gabor2D () ;
   void transform ( int nrows , int ncols , double *img ,
                    int rcent , int ccent , double *rt , double *it ) ;

private:
   double sig ;     // Sigma
   double dt ;      // Delta_t
   double df ;      // Delta_f
   int hl ;         // Filter half length (=halflen)
   int nf ;         // Number of frequencies resolved (=nfreqs)
   int nd ;         // Number of orientation directions (=ndir)
   int neach ;      // Number of coefs for each filter (freq and orient)
   double f0 ;      // Frequency increment = Nyquist / (nfreqs-1)
   double *coefs ;  // FIR coefficients for all frequencies
} ;

/*
--------------------------------------------------------------------------------

   Constructor, destructor

--------------------------------------------------------------------------------
*/

Gabor2D::Gabor2D (
   double *sigma ,     // Data window scale factor
   double *delta_t ,   // Radius of time-domain window
   double *delta_f ,   // Radius of frequency-domain window
   int nfreqs ,        // Number of frequecies, zero through Nyquist
   int ndir ,          // Number of orientation directions
   double orient ,     // Angle (0-180) of first orientation
   int *halflen ,      // Filter half-length (skip this in all direcs)
   int *ok             // Memory allocation ok?
   )
{
   int ifreq, idir, ix, iy ;
   double freq, direc, weight, fac, gconst, *cptr, cosine, sine, p, dsq ;

   *ok = 0 ;               // Start out pessimistic
   orient *= pi / 180.0 ;  // Convert degrees to radians
   nd = ndir ;             // Number of orientation directions
   nf = nfreqs ;           // Number of frequencies
   f0 = 0.5 / (nf - 1) ;   // Increment for successive freqs

/*
   Compute (as needed) sigma, delta_t, and delta_f
*/

   if (*sigma <= 0.0) {
      if (*delta_t <= 0.0) {
         if (*delta_f <= 0.0)
            return ; // Error
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
   Each of the nfreqs * ndir filters will be stored with the center row
   first, then the next row up, et cetera.  The bottom half is not stored
   because it can be computed by symmetry.  Each row of a filter will have
   2*hl+1 coefficients, and there will be hl+1 rows stored.  The real part
   of each coefficient will be followed by the imaginary part.
*/

   *halflen = hl = 1 + 7.4 * sig ;  // Goes to about 1.e-12
   neach = 2 * (hl+1) * (2*hl+1) ;  // Doubles in each filter

   coefs = (double *) malloc ( neach * nf * ndir * sizeof(double) ) ;
   if (coefs == NULL)
      return ;
   *ok = 1 ;

/*
   Compute the filter coefficients.
*/

   gconst = 1.0 / (sig * rootpi) ;        // Normalizer for Gaussian

   fac = -0.5 / (sig * sig) ;             // Common factor in exponent

   cptr = coefs ;                         // Point to coef area
   for (ifreq=0 ; ifreq<nf ; ifreq++) {   // Covers 0 to Nyquist
      freq = ifreq * f0 * two_pi ;        // Freq term in complex exponential
      for (idir=0 ; idir<ndir ; idir++) { // Each orientation direction
         direc = orient + idir * pi / ndir ;
         cosine = cos ( direc ) ;
         sine = sin ( direc ) ;
         for (iy=0 ; iy<=hl ; iy++) {                // Only do top half
            for (ix=-hl ; ix<=hl ; ix++) {           // Do entire row
               p = ix * cosine + iy * sine ;         // Distance along direc
               dsq = ix * ix  +  iy * iy ;           // Distance from center
               weight = gconst * exp ( fac * dsq ) ; // Data window
               *cptr++ = cos ( p * freq ) * weight ; // Real part of filter
               *cptr++ = sin ( p * freq ) * weight ; // And imaginary part
               }
            }
         }
      }
}

Gabor2D::~Gabor2D ()
{
   if (coefs != NULL)
      free ( coefs ) ;
}

void Gabor2D::transform (
   int nrows ,   // Number of rows in image
   int ncols ,   // And columns
   double *img , // Image
   int rcent ,   // Row (origin 0) where filter centered
   int ccent ,   // And column
   double *rt ,  // Real output
   double *it    // Imaginary output
   )
{
   int ifreq, idir, row, col ;
   double *cptr, *imgptr_up, *imgptr_dn, rsum, isum ;

/*
   Make sure we do not overrun the image with the filter.
   Recompute img to point to the center pixel in the filtered span.
*/

   if ((rcent < hl)  ||  (rcent >= nrows-hl))
      return ;
   if ((ccent < hl)  ||  (ccent >= ncols-hl))
      return ;

   img += rcent * ncols + ccent ;

/*
   Apply the filters.  Remember that an element in the bottom half of the
   filter is the complex conjugate of the element reflected along both axes.
   Therefore we can implement the full filter very easily by summing left-
   to-right for the top half and right-to-left (flipping the sine sign) for
   the bottom half.
*/

   cptr = coefs ;                                  // All coefs here
   for (ifreq=0 ; ifreq<nf ; ifreq++) {            // Do all frequencies
      for (idir=0 ; idir<nd ; idir++) {            // Do all orientations
         rsum = isum = 0.0 ;                       // Will cumulate here
         for (row=0 ; row<=hl ; row++) {           // Only saved top half coefs
            imgptr_up = img - row * ncols - hl ;   // Rows above center row
            imgptr_dn = img + row * ncols + hl ;   // And those below, reversed
            if (row) {                             // Do both rows at once
               for (col=-hl ; col<=hl ; col++) {   // Do all columns
                  rsum += *cptr * *imgptr_up ;     // Row above center
                  rsum += *cptr++ * *imgptr_dn ;   // And below
                  isum += *cptr * *imgptr_up++ ;   // Ditto for imaginary part
                  isum -= *cptr++ * *imgptr_dn-- ; // Sign flips below center
                  }
               }
            else {                                 // Do center row alone
               for (col=-hl ; col<=hl ; col++) {   // All columns
                  rsum += *cptr++ * *imgptr_up ;
                  isum += *cptr++ * *imgptr_up++ ;
                  }
               }
            }
         *rt++ = rsum ;  // Output transform for this freq and dir
         *it++ = isum ;
         }
      }
}

