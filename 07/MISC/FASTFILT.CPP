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

   FASTFILT

--------------------------------------------------------------------------------
*/

void fastfilt (
   int n ,          // Length of input/output series
   double *xr ,     // Raw input, in-phase filtered output
   double *xi ,     // In-quadrature filtered output
   double freq ,    // Center frequency, 0-.5
   double s         // Scale (width) factor in frequency domain
   )

{
   int i, halfn ;
   double f, dist, wt ;

/*
   Pack the real series for the short-cut FFT method
*/

   halfn = n / 2 ;               // We only need to transform half n
   for (i=0 ; i<halfn ; i++) {   // Because we take advantage of real input
      xr[i] = xr[2*i] ;          // Pack even terms in real part
      xi[i] = xr[2*i+1] ;        // And odd in imaginary part
      }
   real_fft ( halfn , xr , xi ) ;

/*
   Unpack the transformed real array.  It is conjugate symmetric.
   Recall that the (real) Nyquist point was placed in the imaginary
   DC point, which is actually zero.
   While we are at it, apply the frequency-domain filter to the points
   above the Nyquist frequency.  Simultaneously zero the points to the
   left, as well as the DC and Nyquist points.
   Also, there is conjugate symmetry, so we should flip the sign of the
   imaginary parts to the right of the Nyquist point.  But when we do the
   inverse transform later we would just flip it again.  So avoid that
   double flip.  Similarly, include in wt the factor of 1/n that we
   would impose later for the inverse transform.
*/

   for (i=1 ; i<halfn ; i++) {
      f = (double) i / (double) n ; // Frequency of this point
      dist = (f - freq) / s ;       // Weighted distance from center freq
      wt = exp ( - dist * dist ) ;  // Gaussian weighting function
      wt *= 2.0 ;              // Real and imaginary parts reinforce
      wt /= (double) n ;       // Include this factor for the inverse transform
      xr[n-i] = wt * xr[i] ;   // Since input was real, transform is symmetric
      xi[n-i] = wt * xi[i] ;   // Conjugate symmetry, but avoid double flip
      xr[i] = 0.0 ;            // Zero out left part so that we get
      xi[i] = 0.0 ;            // Both in-phase and in-quadrature outputs
      }
   xr[0] = xi[0] = xr[halfn] = xi[halfn] = 0.0 ;  // Also zero these

/*
   Do the inverse Fourier transform.  Note that the first step, flipping
   the sign of the imaginary parts, was implicitly done above when we
   did not flip them for conjugate symmetry.  Also, the division by n
   was done above in conjunction with the Gaussian weighting.  So we
   can avoid the first two steps!  Just go ahead with the FFT.  Then
   do the final step of flipping imaginary signs.
*/

   fft ( n , xr , xi ) ;
   for (i=0 ; i<n ; i++)
      xi[i] = -xi[i] ;
}

/*
--------------------------------------------------------------------------------

   FastFilter

--------------------------------------------------------------------------------
*/

class FastFilter {

   public:
      FastFilter ( int npts , double *signal ) ;
      ~FastFilter () ;
      void filter ( double freq , double s , int decim ,
                    double *real , double *imag ) ;

   private:
      int n ;
      double *xr, *xi ;
   } ;

/*
   Constructor, destructor
*/

FastFilter::FastFilter (
   int npts ,        // Length of input signal (power of 2)
   double *signal    // Input signal
   )
{
   int i, halfn ;
   double wt ;

   n = 0 ;         // Cheap insurance
   xr = NULL ;     // So destructor does not err

   if (npts <= 0)  // Protect from careless user
      return ;

   xr = (double *) malloc ( npts * sizeof(double) ) ;
   if (xr == NULL)
      return ;

   n = npts ;
   halfn = n / 2 ;
   xi = xr + halfn ;

/*
   Pack the real series for the short-cut FFT method.
   Also include the factor of two that results from the in-phase and
   in-quadrature filters reinforcing, and 1/n for the inverse transform.
   Doing it once now saves doing it every time 'filter' is called later.
*/

   wt = 2.0 / (double) n ;
   for (i=0 ; i<halfn ; i++) {
      xr[i] = wt * signal[2*i] ;
      xi[i] = wt * signal[2*i+1] ;
      }
   real_fft ( halfn , xr , xi ) ;
}

FastFilter::~FastFilter ()
{
   if (xr != NULL)
      free ( xr ) ;
}

/*
   This is the filter function
*/

void FastFilter::filter (
   double freq ,   // Center frequency, 0-.5
   double s ,      // Scale (width) factor in frequency domain
   int decim ,     // Decimation ratio (power of 2)
   double *real ,  // Real output (in-phase)
   double *imag    // Imaginary output (in-quadrature)
   )
{
   int i, newn, nn ;
   double f, dist, wt ;

/*
   The constructor provided us with the packed transform of the input (xr, xi).
   Copy it to the output area, unpacking and removing the center if
   there is any decimation.  Recall that the DC and Nyquist terms, as
   well as the entire left half (below the Nyquist) of this data is zero.
   So we only fill in the right half.  Also, normal unpacking demands that
   we flip the signs of the imaginary parts in the right half.  But the
   first step in the inverse transform is to flip those same signs, so
   avoid the double flip by not doing it at all.
   We might as well apply the Gaussian function too.
*/

   nn = newn = n / decim ;

   for (i=1 ; i<newn/2 ; i++) {
      f = (double) i / (double) n ; // Frequency of this point
      dist = (f - freq) / s ;       // Weighted distance from center freq
      wt = exp ( - dist * dist ) ;  // Gaussian weighting function
      real[--nn] = wt * xr[i] ;
      imag[nn] = wt * xi[i] ;
      }

   while (nn--)
      real[nn] = imag[nn] = 0.0 ;

/*
   Do the inverse Fourier transform.  Note that the first step, flipping
   the sign of the imaginary parts, was implicitly done above when we
   did not flip them for conjugate symmetry.  Also, the division by n
   was done in the constructor.  So we can avoid the first two steps!
   Just go ahead with the FFT.  Then do the final step of flipping
   imaginary signs.
*/

   fft ( newn , real , imag ) ;
   for (i=0 ; i<newn ; i++)
      imag[i] = -imag[i] ;
}


/*
--------------------------------------------------------------------------------

   FastFilterImage

--------------------------------------------------------------------------------
*/

class FastFilterImage {

   public:
      FastFilterImage ( int nrows , int ncols , double *image ) ;
      ~FastFilterImage () ;
      int ok () ;
      void filter ( double freq , double f_width , double angle ,
                    double a_width , int rdec , int cdec ,
                    double *real , double *imag ) ;


   private:
      int nr, nc ;        // Size of image
      double *xr, *xi ;   // Transform saved here
      double *work ;      // Used by constructor (nr) and filter (2nr)
   } ;

/*
   Constructor, destructor
*/

FastFilterImage::FastFilterImage (
   int nrows ,   // Number of rows in image
   int ncols ,   // And columns
   double *img   // Image
   )
{
   int n, ic, ir, r ;
   double *wr, *wi, wt ;

   nr = nc = 0 ;            // Cheap insurance
   xr = xi = work = NULL ;  // So destructor does not err

   if (nrows <= 0  ||  ncols <= 0)  // Protect from careless user
      return ;

   n = (nrows / 2 + 1) * ncols ;  // Size of (almost) unique transform
   xr = (double *) malloc ( 2 * n * sizeof(double) ) ;
   if (xr == NULL)
      return ;

   work = (double *) malloc ( 2 * nrows * sizeof(double) ) ;
   if (work == NULL) {
      free ( xr ) ;
      xr = NULL ;
      return ;
      }

   xi = xr + n ;    // Saves a call to malloc

   nr = nrows ;
   nc = ncols ;

/*
   Copy each column of the block to the work vector, transform it,
   then copy the result to the corresponding column of the output.
   We split the temporary work vector in half, using the first
   half for the real part of the transform, and the second half for
   the imaginary part.  Recall that REAL_FFT wants the even terms of
   the input vector in the real part, and the odd terms in the
   imaginary part.  Also recall that REAL_FFT returns the real part
   of the Nyquist term in wi[0], which is really zero.  The imaginary
   part of the Nyquist term is also zero.

   Also include the factor of two that results from the in-phase and
   in-quadrature filters reinforcing, and 1/(nr*nc) for the two
   inverse Fourier transforms that will be done for the filtering.
   Doing it once now saves doing it every time 'filter' is called later.
*/

   wr = work ;           // Work vector
   wi = work + nr / 2 ;  // Use another part for imaginary
   wt = 2.0 / ((double) nr * (double) nc) ; // Reinforcement, inverse transform

   for (ic=0 ; ic<nc ; ic++) {            // Do each column separately

      for (ir=0 ; ir<nr/2 ; ir++) {       // Copy all rows 2 at a time
         r = 2 * ir ;                     // Even row in block
         wr[ir] = wt * img[r*nc+ic] ;     // Even part into real vector
         ++r ;                            // Odd row in block
         wi[ir] = wt * img[r*nc+ic] ;     // Odd part into imaginary
         }

      real_fft ( nr/2 , wr , wi ) ;       // Transform column ic

      for (ir=0 ; ir<nr/2 ; ir++) {       // Because the input was real,
         xr[ir*nc+ic] = wr[ir] ;          // there are nr/2+1 unique terms.
         xi[ir*nc+ic] = wi[ir] ;          // The neg freqs are just conjugates
         }                                // and so are ignored.
      xr[nr/2*nc+ic] = wi[0] ;            // Returned real part of Nyquist here
      xi[nr/2*nc+ic] = xi[ic] = 0.0 ;     // These are really zero
      } // For all columns

/*
   At this time, all nc columns of the block have been transformed.
   They are now in place in the private vectors "xr" and "xi".
   The final step is to transform each of the nr/2+1 rows.
   The first and last rows are entirely real, so if we were terribly
   concerned with efficiency, we could use the real transform routine
   REAL_FFT to do it. But the relatively small savings in time is rarely
   worth the increase in complexity of the code, so we will use the
   full complex routine.  Of course, the interior rows all need the full
   complex routine, as they have no special properties.
*/

   for (ir=0 ; ir<=nr/2 ; ir++)
      fft ( nc , xr+ir*nc , xi+ir*nc ) ;
}

int FastFilterImage::ok ()
{
   return nr != 0 ;
}

FastFilterImage::~FastFilterImage ()
{
   if (xr != NULL)
      free ( xr ) ;
   if (work != NULL)
      free ( work ) ;
}

/*
--------------------------------------------------------------------------------

   This local function is the weight function used for the frequency-domain
   filter.  It could be changed as desired.  For example, a symmetric
   Gaussian shape would implement the basic Gabor / Morlet wavelet.
   Here, we use the circular sector approach discussed in the text.

--------------------------------------------------------------------------------
*/

static double fcent, acent, fwt, awt ;
static double pi = 3.141592653589793 ;
static double two_pi = 2.0 * pi ;
static double half_pi = 0.5 * pi ;

static void weight_init (
   double freq ,    // Center frequency, 0-.5
   double f_width , // Half-amplitude frequency difference
   double angle ,   // CW Direction in radians
   double a_width   // Half-amplitude angle difference
   )
{
   fcent = freq ;
   if (angle <= pi)
      acent = angle ;
   else 
      acent = angle - two_pi ;
   fwt = log ( 0.5 ) / (f_width * f_width) ;
   awt = log ( 0.5 ) / (a_width * a_width) ;
}

static double weight ( double xfreq , double yfreq )
{
   double fdist, angle, adist ;

   fdist = sqrt ( xfreq * xfreq + yfreq * yfreq ) - fcent ;
   angle = atan2 ( -yfreq , -xfreq ) ;
   adist =  fabs ( angle - acent ) ;
   if (adist > pi)                // Wrapping around the discontinuity
      adist = two_pi - adist ;    // would cause error
   if (adist > half_pi)           // Keep response on half-plane
      return 0.0 ;                // so we get a quadrature pair
   return exp ( fwt * fdist * fdist  +  awt * adist * adist ) ;
}

/*
--------------------------------------------------------------------------------

   This is the filter function

--------------------------------------------------------------------------------
*/

void FastFilterImage::filter (
   double freq ,    // Center frequency, 0-.5
   double f_width , // Half-amplitude frequency difference
   double angle ,   // CW Direction in radians
   double a_width , // Half-amplitude angle difference
   int rdec ,       // Row decimation ratio (power of 2)
   int cdec ,       // And column
   double *real ,   // Real output (in-phase)
   double *imag     // Imaginary output (in-quadrature)
   )
{
   int irow, icol, newnr, newnc ;
   double xfreq, yfreq, wt, rnyq, inyq, *wr, *wi ;

   weight_init ( freq , f_width , angle , a_width ) ;

   newnr = nr / rdec ;
   newnc = nc / cdec ;

/*
   First, do the upper half, which is nothing more than a weighted copy
   of the stored transform.  Do the DC and all positive vertical frequencies
   below the Nyquist.  For each, do all horizontal frequencies.
*/

   for (irow=0 ; irow<newnr/2 ; irow++) {      // Nonneg Y freqs below Nyquist
      yfreq = (double) irow / (double) nr ;    // Vertical frequency
      for (icol=0 ; icol<newnc/2 ; icol++) {   // All X freqs except Nyquist
         xfreq = (double) icol / (double) nc ; // Positive Horizontal freq
         wt = weight ( xfreq , yfreq ) ;
         real[irow*newnc+icol] = wt * xr[irow*nc+icol] ;
         imag[irow*newnc+icol] = wt * xi[irow*nc+icol] ;
         if (! icol)        // DC has no corresponding negative
            continue ;      // So skip the negative frequency stuff below
         wt = weight ( -xfreq , yfreq ) ;
         real[irow*newnc+newnc-icol] = wt * xr[irow*nc+nc-icol] ;
         imag[irow*newnc+newnc-icol] = wt * xi[irow*nc+nc-icol] ;
         }
      // X Nyquist is assumed zero if decimating, else just conjugate
      if (cdec > 1)
         rnyq = inyq = 0.0 ;
      else {
         rnyq = xr[irow*nc+nc/2] ;
         inyq = xi[irow*nc+nc/2] ;
         }
      wt = weight ( 0.5 / (double) cdec , yfreq ) ;
      real[irow*newnc+newnc/2] = wt * rnyq ;
      imag[irow*newnc+newnc/2] = wt * inyq ;
      }

/*
   Now do the lower half.  This is negative vertical frequencies.
   We get the values by using the conjugate of the stored transform
   at the opposite diagonal position (negative of the frequency).
*/

   for (irow=1 ; irow<newnr/2 ; irow++) {      // Negative Y freqs
      yfreq = - (double) irow / (double) nr ;  // Vertical frequency
      for (icol=0 ; icol<newnc/2 ; icol++) {   // All X freqs except Nyquist
         xfreq = (double) icol / (double) nc ; // Nonnegative Horizontal freq
         wt = weight ( xfreq , yfreq ) ;
         if (! icol) {   // DC is a simple weighted copy
            real[(newnr-irow)*newnc] = wt * xr[irow*nc] ;
            imag[(newnr-irow)*newnc] = -wt * xi[irow*nc] ;
            continue ;
            }
         // Get positive X frequencies from right side of source
         real[(newnr-irow)*newnc+icol] = wt * xr[irow*nc+nc-icol] ;
         imag[(newnr-irow)*newnc+icol] = -wt * xi[irow*nc+nc-icol] ;
         // Get negative X frequencies from left side of source
         wt = weight ( -xfreq , yfreq ) ;
         real[(newnr-irow)*newnc+newnc-icol] = wt * xr[irow*nc+icol] ;
         imag[(newnr-irow)*newnc+newnc-icol] = -wt * xi[irow*nc+icol] ;
         }
      // X Nyquist is assumed zero if decimating, else just conjugate
      if (cdec > 1)
         rnyq = inyq = 0.0 ;
      else {
         rnyq = xr[irow*nc+nc/2] ;
         inyq = -xi[irow*nc+nc/2] ;
         }
      wt = weight ( 0.5 / (double) cdec , yfreq ) ;
      real[(newnr-irow)*newnc+newnc/2] = wt * rnyq ;
      imag[(newnr-irow)*newnc+newnc/2] = wt * inyq ;
      }

/*
   Now do the vertical (Y) Nyquist frequency
*/

   if (rdec == 1) {
      for (icol=0 ; icol<newnc/2 ; icol++) {   // All X freqs except Nyquist
         xfreq = (double) icol / (double) nc ; // Positive Horizontal freq
         wt = weight ( xfreq , 0.5 ) ;
         real[newnr/2*newnc+icol] = wt * xr[nr/2*nc+icol] ;
         imag[newnr/2*newnc+icol] = wt * xi[nr/2*nc+icol] ;
         if (! icol)        // DC has no corresponding negative
            continue ;      // So skip the negative frequency stuff below
         wt = weight ( -xfreq , 0.5 ) ;
         real[newnr/2*newnc+newnc-icol] = wt * xr[nr/2*nc+nc-icol] ;
         imag[newnr/2*newnc+newnc-icol] = wt * xi[nr/2*nc+nc-icol] ;
         }
      // X Nyquist is assumed zero if decimating, else just conjugate
      if (cdec > 1)
         rnyq = inyq = 0.0 ;
      else {
         rnyq = xr[nr/2*nc+nc/2] ;
         inyq = xi[nr/2*nc+nc/2] ;
         }
      wt = weight ( 0.5 / (double) cdec , 0.5 ) ;
      real[newnr/2*newnc+newnc/2] = wt * rnyq ;
      imag[newnr/2*newnc+newnc/2] = wt * inyq ;
      }

   else {  // Case of row decimation
      for (icol=0 ; icol<newnc ; icol++)
         real[newnr/2*newnc+icol] = imag[newnr/2*newnc+icol] = 0.0 ;
      }

/*
   Do the inverse Fourier transform.  Note that the division by n squared
   was done in the constructor.  So we can avoid that now.
*/

   wr = work ;
   wi = work + newnr ;

   for (icol=0 ; icol<newnc ; icol++) {

      for (irow=0 ; irow<newnr ; irow++) {
         wr[irow] = real[irow*newnc+icol] ;
         wi[irow] = -imag[irow*newnc+icol] ;
         }

      fft ( newnr , wr , wi ) ;

      for (irow=0 ; irow<newnr ; irow++) {
         real[irow*newnc+icol] = wr[irow] ;
         imag[irow*newnc+icol] = wi[irow] ;
         }
      }

   for (irow=0 ; irow<newnr ; irow++) {
      fft ( newnc , real+irow*newnc , imag+irow*newnc ) ;
      for (icol=0 ; icol<newnc ; icol++)
         imag[irow*newnc+icol] = -imag[irow*newnc+icol] ;
      }
}
