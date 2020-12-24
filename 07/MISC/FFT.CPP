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

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define PI 3.141592653589793

static void bit_reverse ( int n , double *xr , double *xi ) ;
static void butterflies ( int n , double *xr , double *xi ) ;

void fft (
   int n ,       // Length of vectors
   double *xr ,  // Real components
   double *xi    // Imaginary components
   )
{
   bit_reverse ( n , xr , xi ) ;
   butterflies ( n , xr , xi ) ;
}

static void bit_reverse ( int n , double *xr , double *xi )
{
   int i, bitrev, k, half_n ;
   double temp ;

   bitrev = 0 ;     // Will count in bit-reversed order
   half_n = n / 2 ;

   for (i=1 ; i<n ; i++) {   // Do every element

/*
   Increment bit-reversed counter
*/

      ++bitrev ;
      k = half_n ;
      while (bitrev > k) {
         bitrev -= k ;
         k >>= 1 ;
         }
      bitrev += k-1 ;

/*
   Swap straight-counter element with bit-reversed element (just once!)
*/

      if (i < bitrev) {
         temp = xr[i] ;
         xr[i] = xr[bitrev] ;
         xr[bitrev] = temp ;
         temp = xi[i] ;
         xi[i] = xi[bitrev] ;
         xi[bitrev] = temp ;
         }
      }
}

static void butterflies ( int n , double *xr , double *xi )
{
   int i, m, mmax, step, other ;
   double theta, tr, ti, wr, wi, wkr, wki ;

   for (mmax=1 ; mmax<n ; mmax=step) {
      step = 2 * mmax ;

      wr = 1.0 ;
      wi = 0.0 ;
      theta = PI / (double) mmax ;
      tr = sin ( 0.5 * theta ) ;
      wkr = -2.0 * tr * tr ;
      wki = sin ( theta ) ;

      for (m=0 ; m<mmax ; m++) {
         for (i=m ; i<n ; i+=step) {
            other = i + mmax ;
            tr = wr * xr[other]  -  wi * xi[other] ;
            ti = wr * xi[other]  +  wi * xr[other] ;
            xr[other] = xr[i] - tr ;
            xi[other] = xi[i] - ti ;
            xr[i] += tr ;
            xi[i] += ti ;
            }
         tr = wr ;
         wr += tr * wkr  -  wi * wki ;
         wi += tr * wki  +  wi * wkr ;
         }
      }
}

void real_fft (
   int n ,       // Length of each vector (Total is 2n)
   double *xr ,  // In: 0,2,4,... Out:Real parts
   double *xi    // In: 1,3,5,... Out: Imaginary parts
   )
{
   int i, j ;
   double theta, wr, wi, wkr, wki, t, h1r, h1i, h2r, h2i ;

   fft ( n , xr , xi ) ;

/*
   Use the guaranteed zero xi[0] to actually return xr[n]
*/

   t = xr[0] ;
   xr[0] = t + xi[0] ;
   xi[0] = t - xi[0] ;

/*
   Now do the remainder through n-1
*/

   theta = PI / (double) n ;
   t = sin ( 0.5 * theta ) ;
   wr = 1.0 + (wkr = -2.0 * t * t) ;
   wi = wki = sin ( theta ) ;

   for (i=1 ; i<n/2 ; i++) {
      j = n - i ;
      h1r =  0.5 * (xr[i] + xr[j]) ;
      h1i =  0.5 * (xi[i] - xi[j]) ;
      h2r =  0.5 * (xi[i] + xi[j]) ;
      h2i = -0.5 * (xr[i] - xr[j]) ;
      xr[i] =  wr * h2r  -  wi * h2i  +  h1r ;
      xi[i] =  wr * h2i  +  wi * h2r  +  h1i ;
      xr[j] = -wr * h2r  +  wi * h2i  +  h1r ;
      xi[j] =  wr * h2i  +  wi * h2r  -  h1i ;
      t = wr ;
      wr += t * wkr  -  wi * wki ;
      wi += t * wki  +  wi * wkr ;
      }
}

/*
--------------------------------------------------------------------------------

   Normalization routines

--------------------------------------------------------------------------------
*/

void keep_harmonics (
   int *n ,      // Length of input/output vectors
   int p ,       // Periods in sample
   double *xr ,  // Real components
   double *xi    // Imaginary components
   )
{
   int i ;

   if (*n % p) {
      printf ( "\nERROR: n must be a multiple of p" ) ;
      return ;
      }

   *n /= p ;
   for (i=1 ; i<*n ; i++) {
      xr[i] = xr[i*p] ;
      xi[i] = xi[i*p] ;
      }
}

void force_clockwise (
   int n ,       // Length of vectors
   double *xr ,  // Real components
   double *xi    // Imaginary components
   )
{
   int i, nyquist ;
   double mag1, magm1, temp ;

   mag1  = xr[1]   * xr[1]    +  xi[1]   * xi[1] ;
   magm1 = xr[n-1] * xr[n-1]  +  xi[n-1] * xi[n-1] ;
   if (mag1 >= magm1)  // If we are tracing clockwise
      return ;         // nothing to fix

   nyquist = (n+1) / 2 ;  // N may be odd (unlikely, though!)
   for (i=1 ; i<nyquist ; i++) { // Reflect about Nyquist
      temp = xr[i] ;
      xr[i] = xr[n-i] ;
      xr[n-i] = temp ;
      temp = xi[i] ;
      xi[i] = xi[n-i] ;
      xi[n-i] = temp ;
      }
}

void normalize_scale (
   int n ,       // Length of vectors
   double *xr ,  // Real components
   double *xi    // Imaginary components
   )
{
   int i ;
   double factor ;

   factor = xr[1] * xr[1]  +  xi[1] * xi[1] ;
   if (factor == 0.0)
      return ;

   factor = 1.0 / sqrt ( factor ) ;
   for (i=1 ; i<n ; i++) {
      xr[i] *= factor ;
      xi[i] *= factor ;
      }
}

void normalize_phase (
   int n ,       // Length of vectors
   double *xr ,  // Real components
   double *xi    // Imaginary components
   )
{
   int i, nyquist ;
   double mag, phase, theta ;

   mag = sqrt ( xr[1] * xr[1]  +  xi[1] * xi[1] ) ;
   if (mag > 0.0)
      theta = atan2 ( xi[1] , xr[1] ) ;
   else 
      theta = 0.0 ;

   nyquist = (n+1) / 2 ;  // N may be odd (unlikely, though!)
   for (i=1 ; i<nyquist ; i++) {

      mag = sqrt ( xr[i] * xr[i]  +  xi[i] * xi[i] ) ;
      if (mag > 0.0)
         phase = atan2 ( xi[i] , xr[i] ) - theta * i ;
      else 
         phase = 0.0 ;
      xr[i] = mag * cos(phase) ;
      xi[i] = mag * sin(phase) ;

      mag = sqrt ( xr[n-i] * xr[n-i]  +  xi[n-i] * xi[n-i] ) ;
      if (mag > 0.0)
         phase = atan2 ( xi[n-i] , xr[n-i] ) + theta * i ;
      else 
         phase = 0.0 ;
      xr[n-i] = mag * cos(phase) ;
      xi[n-i] = mag * sin(phase) ;
      }
}

void normalize_rotation_shift (
   int n ,       // Length of vectors
   int k ,       // Secondary coef to use, 0 to pick max
   double *xr ,  // Real components
   double *xi    // Imaginary components
   )
{
   int i, freq, m, mbest ;
   double mag, phase, theta_1, theta_k, phi, rho, crit, best ;
   double real, imag, offset ;

/*
   If the user specified some k>0 we will use it.
   Otherwise, we pick k as the coefficient having max magnitude.
*/

   if (k <= 0) {  // User wants us to choose k as max
      best = -1.e30 ;
      for (i=2 ; i<n ; i++) {
         mag = xr[i] * xr[i]  +  xi[i] * xi[i] ;
         if (mag > best) {
            best = mag ;
            k = i ;
            }
         }
      } // Choose k

/*
   Compute the phase angles of H(1) and H(k).
*/

   if ((xr[1] != 0.0)  ||  (xi[1] != 0.0))
      theta_1 = atan2 ( xi[1] , xr[1] ) ;
   else 
      theta_1 = 0.0 ;

   if ((xr[k] != 0.0)  ||  (xi[k] != 0.0))
      theta_k = atan2 ( xi[k] , xr[k] ) ;
   else 
      theta_k = 0.0 ;

/*
   Compute the multiplicity, m, and the normalization constants, phi and rho.
*/

   freq = (k <= n/2)  ?  k  :  k-n ;
   m = abs ( freq - 1 ) ;

   phi = (theta_1 - theta_k) / (double) (freq - 1) ;
   rho = (theta_k - freq * theta_1) / (double) (freq - 1) ;

/*
   Apply the normalization, shifting the starting point by phi
   and rotating by rho.  This gives us one of the m normalizations.
   If m is 1, that is all we need, so save the normalized transform
   directly, then return.  Otherwise, temporarily save the magnitude
   in xr and the phase in xi.  Note that for H[1] and H[k] the
   magnitude is the real part and the phase is zero, so those
   two terms will be OK as is!
*/

   for (i=1 ; i<n ; i++) {
      mag = sqrt ( xr[i] * xr[i]  +  xi[i] * xi[i] ) ;

      if ((i == 1)  ||  (i == k)) {  // Can do these explicitly
         xr[i] = mag ;
         xi[i] = 0.0 ;
         continue ;
         }

      freq = (i <= n/2)  ?  i  :  i-n ;
      if (mag > 0.0)
         phase = atan2 ( xi[i] , xr[i] )  +  freq * phi + rho ;
      else 
         phase = 0.0 ;

      if (m == 1) {  // If only 1 possibility just do it
         xr[i] = mag * cos(phase) ;
         xi[i] = mag * sin(phase) ;
         }
      else {         // If many, stay in polar for now
         xr[i] = mag ;
         xi[i] = phase ;
         }
      }

   if (m == 1)  // If this was the only possibility
      return ;  // then we are done

/*
   We have found one of the m normalizations that zero the phase
   of H(1) and H(k).  Each of the others is offset by equal
   portions of a full circle.  Try them all, keeping track of
   the one that has maximum positive real energy.
   Recall that we have stored the magnitude in xr and the
   phase in xi to avoid going back to polar each time.
*/

   offset = 2.0 * PI / (double) m ;  // Each is offset this amount
   mbest = -1 ;  // Keep track of best here

   while (--m >= 0) {  // Try each of the m possibilities
      crit = 0.0 ;     // Will sum criterion here
      for (i=2 ; i<n ; i++) { // No need to include constant H(1)
         if (i == k)          // or H(k) in the criterion sum
            continue ;
         freq = (i <= n/2)  ?  i  :  i-n ;
         real = xr[i] * cos( xi[i]  +  offset * m * (freq-1) ) ;
         crit += real * abs ( real ) ;
         }
      if ((mbest < 0)  ||  (crit > best)) {
         best = crit ;
         mbest = m ;
         }
      }

/*
   We now know which of the m normalizations was best.  Apply it.
   Do not bother with H(1) and H(k), as they are already correct.
*/

   offset *= mbest ;  // This is the best angle offset

   for (i=2 ; i<n ; i++) {
      if (i == k)
         continue ;
      freq = (i <= n/2)  ?  i  :  i-n ;
      real = xr[i] * cos( xi[i]  +  offset * (freq-1) ) ;
      imag = xr[i] * sin( xi[i]  +  offset * (freq-1) ) ;
      xr[i] = real ;
      xi[i] = imag ;
      }
}

/*
--------------------------------------------------------------------------------

   2-D FFT

--------------------------------------------------------------------------------
*/

void fft2dr (
   unsigned char *image , // Input image, returned intact
   int nrows ,            // Number of rows in image
   int ncols ,            // And columns
   int row ,              // Upper left pixel (origin 0) of
   int col ,              // the block to be transformed
   int nr ,               // Size of the transformed
   int nc ,               // block (must be power of 2!)
   double *work ,         // Work vector nr long
   double *real ,         // nr/2+1 (rows) by nc (columns)
   double *imag           // output matrix of transform
   )
{
   int ir, ic, r, c ;
   double *wr, *wi, rcent, ccent, rden, cden, rdist, cdist, w ;

/*
   Make sure the user did not position the block so that it extends
   outside the image.
*/

   if ((row < 0)  ||  (row > nrows-nr)  ||  (col < 0)  ||  (col > ncols-nc))
      return ;

   rcent = 0.5 * (nr - 1.0) ;  // Center of block for the
   ccent = 0.5 * (nc - 1.0) ;  // Welch data window
   rden = 0.5 * (nr + 1.0) ;   // Denominators for that window
   cden = 0.5 * (nc + 1.0) ;

/*
   Copy each column of the block to the work vector, transform it,
   then copy the result to the corresponding column of the output.
   We split the user-supplied work vector in half, using the first
   half for the real part of the transform, and the second half for
   the imaginary part.  Recall that REAL_FFT wants the even terms of
   the input vector in the real part, and the odd terms in the
   imaginary part.  Also recall that REAL_FFT returns the real part
   of the Nyquist term in wi[0], which is really zero.  The imaginary
   part of the Nyquist term is also zero.
*/

   wr = work ;           // User-supplied work vector nr long
   wi = work + nr / 2 ;  // Use second half for imaginary part

   for (ic=0 ; ic<nc ; ic++) {            // Do each column separately
      c = col + ic ;                      // Column in image
      cdist = (ic - ccent) / cden ;       // Column distance for Welch window
      cdist *= cdist ;                    // Work with squared distance

      for (ir=0 ; ir<nr/2 ; ir++) {       // Copy all rows 2 at a time
         r = row + 2 * ir ;               // Even row in block
         rdist = (2 * ir - rcent) / rden ;// Row distance for Welch window
         rdist *= rdist ;                 // Work with squared distance
         w = 1.0 - rdist - cdist ;        // This is the Welch window
         if (w < 0.0)                     // It must range from 0 to 1
            w = 0.0 ;                     // This cuts off corners though!
         wr[ir] = w * image[r*ncols+c] ;  // Even part into real vector
         ++r ;                            // Odd row in block
         rdist = (2 * ir + 1 - rcent) / rden ;// Row distance for Welch window
         rdist *= rdist ;                 // Work with squared distance
         w = 1.0 - rdist - cdist ;        // This is the Welch window
         if (w < 0.0)                     // It must range from 0 to 1
            w = 0.0 ;                     // This cuts off corners though!
         wi[ir] = w * image[r*ncols+c] ;  // Odd part into imaginary
         }

      real_fft ( nr/2 , wr , wi ) ;       // Transform column ic

      for (ir=0 ; ir<nr/2 ; ir++) {       // Because the input was real,
         real[ir*nc+ic] = wr[ir] ;        // there are nr/2+1 unique terms.
         imag[ir*nc+ic] = wi[ir] ;        // The neg freqs are just conjugates
         }                                // and so are ignored.
      real[nr/2*nc+ic] = wi[0] ;          // Returned real part of Nyquist here
      imag[nr/2*nc+ic] = imag[ic] = 0.0 ; // These are really zero
      } // For all columns

/*
   At this time, all nc columns of the block have been transformed.
   They are now in place in the output vectors "real" and "imag".
   The final step is to transform each of the nr/2+1 rows.
   The first and last rows are entirely real, so if we were terribly
   concerned with efficiency, we could use the real transform routine
   REAL_FFT to do it. But the relatively small savings in time is rarely
   worth the increase in complexity of the code, so we will use the
   full complex routine.  Of course, the interior rows all need the full
   complex routine, as they have no special properties.
*/

   for (ir=0 ; ir<=nr/2 ; ir++)
      fft ( nc , real+ir*nc , imag+ir*nc ) ;
}
