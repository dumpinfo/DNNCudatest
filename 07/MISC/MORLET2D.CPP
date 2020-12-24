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
/*  MORLET2D - Two-dimensional Morlet wavelet transform                       */
/*                                                                            */
/******************************************************************************/

#include <math.h>
#include <stdlib.h>

/*
--------------------------------------------------------------------------------

   Morlet2D - Construct and compute 2D Morlet wavelet transform

   The constructor computes and saves all wavelet coefficients.
   It also returns several parameters that the user may find useful.

   The function "transform" actually performs the transform.  It may be
   called as many times as desired after the constructor has been called.


   >>> The following parameters are in the constructor parameter list:

   shape - The cross-section shape of the mother wavelet is expressed as the
     ratio of its center frequency to the radius of its frequency-domain
     window (which is 0.5 / (sqrt(2) * pi), or about 0.1125).
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
     
   spacing - This many sample points will separate each of the smallest-scale
     wavelets in the spatial domain.  A good value will be near the sample
     rate returned by the constructor.  Smaller values will fail to cover the
     appropriate breadth.  Larger values leave information gaps in space.
     Values greater than about 1.75 times the sample rate result in serious
     information gaps in the spatial domain.  If this is input zero or negative,
     the constructor returns before computing coefficients (so the new object
     must be deleted!).  This allows the user to examine samprate to set an
     appropriate value for spacing.

   nvoices - The number of voices used includes the basic wavelet at each
     scale, so this parameter must be at least one.  Larger values are
     always needed to have a decent frame.  At least 3 or 4 are necessary for
     small values of the shape parameter, with more needed as the shape
     parameter increases.

   nscales - The number of scales includes the mother wavelet, which is the
     smallest scale.  Thus, this parameter must be at least 1.  When actual
     data is transformed, all of these scales wil be computed.  The image
     must be large enough to accomodate them.  This is in contrast to the
     situation with the one-dimensional transform.

   ndir - Number of orientation directions, at least 1.  If more than 1, they
     will be equally spaced around a half circle, with the first aligned
     at 'orient'.

   orient - Orientation angle in degrees (0-180) of the first filter.

   halflen - This is returned as the half-length of the largest-scale filter.
     Its value tells the user the extent of influence of data around each
     spatial-domain center.  It also enables the user to calculate memory
     requirements in case there is a problem.

   border - This is returned as the number of image points that will be 
     referenced around the center point.  It is usually equal to halflen,
     the half-length of the filter for the largest-scale wavelet.
     If the user specifies an unusually large spacing (a high multiple of
     the sample rate) then the border size may be determined by the
     half-length of the second-largest scale filter added to its extreme
     center position.  The transform routine cannot be called with a center
     specified closer to the image border than this many pixels.

   samprate - This is returned as the sample rate: the number of sample
     points per unit spatial distance in the basic wavelet equation.
     The primary reason for returning this value is that rounding it to
     the nearest integer gives a good value for the spatial-domain spacing
     parameter for the transform routine.

   voicefac - The period of each successive voice at a given scale is
     multipled by this returned quantity.  It is the nvoices'th root of two.

   tcents - This is returned as the number of image centers at which filters
     will be applied.  The largest-scale filter will be applied just once,
     centered at the user's center.  The next largest will be applied in
     a 3 by 3 block whose center is at the user's center.  The number of
     centers around the user's center will double with each successively
     smaller scale.  The extreme position of the centers will all be the
     same, while the distance between each will be divided by two for each
     smaller scale.  Thus, the third-largest scale will have a 5 by 5 block
     of centers, the next 9 by 9, et cetera.

   ok - This is returned 1 if all went well, and 0 if there was insufficient
     memory for storing the FIR filter coefficients or if the user specified
     nonsense parameters.


   >>> The following parameters are in the "transform" parameter list:

   nrows - Number of rows in the image

   ncols - Number of columns in the image

   *img - The image.  It must be stored as a vector with the first (top)
     row first, then the second row, et cetera.

   rcent - Row (origin 0) where the filter is to be centered.  It must be
     greater than or equal to 'border'.  It must be  strictly less than
     nrows - border.  This prevents the filter from spilling outside the image.
     If a closer approach to the edges is needed, pad the image with zeros.

   ccent - Column center as above.

   *rt - This is the real part of the output transform, a vector
     tcents * nvoices * ndir doubles long.
     Results for the single center of the largest scale appear first.
     These are followed by the 9 results for the 3 by 3 block at the next
     scale.  Then come the 25 results for the next scale, et cetera.
     All block centers appear starting at the upper-left corner, working
     across first, then down.
     For each filter center, the basic voice appears first, then increasing
     periods follow.  For each voice, the direction at 'orient' appears first,
     followed by the other orientations in counterclockwise rotation.
     In other words, for this output array, the orientation changes fastest,
     then voice, then column center (left to right), then row center (top to
     bottom), and finally scale running from largest to smallest.

   *it - The imaginary part, as above, is here.


   The constructor will need to allocate a work area to hold the FIR filter
   coefficients.
   This is 2 * (halflen+1) * (2*halflen+1) * ndir * nvoices doubles.
   Each coefficient is complex (2 doubles).
   There will be nvoices * ndir filters saved.
   Only the center row and upper half of each will be found.
   Each row is 2*halflen+1 long, and there are halflen+1 rows.
   An element in the bottom half of the filter is the complex conjugate of the
   element reflected along both axes.

--------------------------------------------------------------------------------
*/

static double pi = 3.141592653589793 ;
static double two_pi = 2. * pi ;
static double root2 = sqrt ( 2.0 ) ;
static double rootpi = sqrt ( pi ) ;

class Morlet2D {

public:
   Morlet2D ( double shape , double rate , int spacing , int nvoices ,
              int nscales , int ndir , double orient , int *halflen ,
              int *border , double *samprate , double *voicefac ,
              int *tcents , int *ok ) ;
   ~Morlet2D () ;
   void transform ( int nrows , int ncols , double *img ,
                    int rcent , int ccent , double *rt , double *it ) ;

private:
   double kparam ;  // "Frequency" shape parameter k
   double srate ;   // Sample rate (samples per unit time)
   int nv ;         // Number of voices
   int ns ;         // Number of scales
   int nd ;         // Number of orientation directions (=ndir)
   int space ;      // Distance between centers of shortest filter
   int hl ;         // Longest filter half length (=halflen)
   int bord ;       // Most distant filter extent (=border)
   int nsmall ;     // Number of smallest-scale centers each side
   int neach ;      // Number of coefs for each filter (voice and orient)
   double *coefs ;  // FIR coefficients for all frequencies
} ;

/*
--------------------------------------------------------------------------------

   Constructor, destructor

--------------------------------------------------------------------------------
*/

Morlet2D::Morlet2D (
   double shape ,     // f/delta, Morlet used 7.54
   double rate ,      // Multiple of Nyquist, at least 1.0
   int spacing ,      // Samples between smallest-scale filter centers
   int nvoices ,      // Number of voices, at least 1
   int nscales ,      // Number of scales, at least 1
   int ndir ,         // Number of orientation directions
   double orient ,    // Angle (0-180) of first orientation
   int *halflen ,     // Half-length of longest filter
   int *border ,      // Furthest extent from center (skip this in all direcs)
   double *samprate , // Samples per unit time
   double *voicefac , // Factor by which period is multiplied for voices
   int *tcents ,      // Number of transform centers output
   int *ok            // Memory allocation ok?
   )
{
   int i, idir, ix, iy, iv ;
   double direc, weight, window_radius, fac, vfac, con, gconst, *cptr ;
   double cosine, sine, p, dsq, vgc, dsqmult, vf, fx, fy ;

   orient *= pi / 180.0 ;  // Convert degrees to radians

/*
   Verify that parameters are legal.
   If the shape-determining frequency is very low, normalization is poor
   unless the sample rate is well above the Nyquist limit.
   (This is a crude check, but it inspires more care than no check at all.)
   The specified sample rate must be at least 1.0 times that limit.
   The number of voices and scales must be at least 1.
*/

   *ok = 0 ;               // Start out pessimistic
   coefs = NULL ;          // So destructor doesn't do bad free if failure here
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
   1.e-5 times its max by the end of the filter.  If multiple voices are used,
   remember that we must accomodate the largest voice.
   For each voice, we will store the coefficients for the largest scale member
   of that family.  Smaller scale members will be derived by decimation.
   Thus, the number of coefficients needed for that longest filter is the
   half-length of the shortest filter times the largest scale (2 ** (ns-1)).
   Note that 2 ** (ns-1) / 2 is also the number of smallest-scale filters that
   will be applied in all directions (radius) around the user's center.
*/

   nd = ndir ;             // Number of orientation directions
   ns = nscales ;          // Number of scales
   nv = nvoices ;          // Number of voices
   space = spacing ;       // Distance between centers of shortest filter

   i = ns ;                // Number of scales
   nsmall = 1 ;            // Compute 2 ** (ns-1)
   while (--i)             // which is twice the number of smallest-scale
      nsmall *= 2 ;        // filters on each side of the user's center point

   hl = 1 + 4.8 * srate * pow ( 2.0 , (double) (nv-1) / (double) nv ) ;
   *halflen = hl = hl * nsmall ; // Half-length of longest filter

   if (space <= 0)         // Abort if user just wants to examine parameters
      return ;

/*
   Normally, the extent of the image needed (border) will be the half-length
   of the longest filter.  But if the user chooses to sacrifice frame
   quality by specifying a large 'spacing' parameter, this max extent may be
   determined by the extent of the second-largest scale filter, because its
   center is out at the limit.
   Also, we are now done using nsmall to hold 2 ** (ns-1), so divide it by
   2 to get the true value of nsmall, which will be 0 if ns=1.
*/

   nsmall /= 2 ;

   if ((space * nsmall + hl / 2)  >  hl)  // Extent of second-largest filter
      *border = bord = space * nsmall + hl / 2 ;
   else                                   // Extent of largest
      *border = bord = hl ;

/*
   Compute tcents, the number of transform centers that will be output
   by the transform routine.  The user will have to allocate rt and it
   for tcents * nvoices * ndir doubles.
   The largest-scale wavelet is evaluated at one center.  The next scale
   is evaluated at the center, plus one on each side.  The next has the
   center plus two on each side, et cetera.
*/

   *tcents = 1 ;             // Largest-scale wavelet at center only
   i = ns ;                  // Number of scales
   idir = 1 ;                // Number of centers on each side
   while (--i) {             // Sum for each additional scale
      idir *= 2 ;            // Side count doubles for each smaller scale
      *tcents += (idir+1) * (idir+1) ; // Both sides plus center
      }

/*
   Each of the nvoices * ndir filters will be stored with the center row
   first, then the next row up, et cetera.  The bottom half is not stored
   because it can be computed by symmetry.  Each row of a filter will have
   2*hl+1 coefficients, and there will be hl+1 rows stored.  The real part
   of each coefficient will be followed by the imaginary part.
*/

   neach = 2 * (hl+1) * (2*hl+1) ;  // Doubles in each filter

   coefs = (double *) malloc ( neach * nv * nd * sizeof(double) ) ;
   if (coefs == NULL)
      return ;
   *ok = 1 ;

/*
   Compute the filter coefficients.
   The sample rate for a voice is the smallest-scale sample rate, srate,
   times the scale of the largest-scale member (since we compute and save
   only its coefs) times the rate factor for that voice.
   The unit time change per point is the reciprocal of that sample rate.
*/

   fac = 1.0 / (srate * pow ( 2.0 , (double) (ns-1))) ;// Largest-scale rate
   gconst = fac / rootpi ;
   con = exp ( -0.5 * kparam * kparam ) ;        // Centering constant

   for (iv=0 ; iv<nv ; iv++) {     // For all voices
      vfac = pow ( 2.0 , -(double) iv / (double) nv ) ; // Voice factor
      vgc = vfac * vfac * gconst ;
      vf = vfac * fac ;
      dsqmult = vf * vf ;
      cptr = coefs + iv * neach * nd ;    // Point to this voice's coef area
      for (idir=0 ; idir<nd ; idir++) { // Each orientation direction
         direc = orient + idir * pi / nd ;
         cosine = cos ( direc ) ;
         sine = sin ( direc ) ;
         for (iy=0 ; iy<=hl ; iy++) {                // Only do top half
            fy = iy ;                                // Convert to fpt
            for (ix=-hl ; ix<=hl ; ix++) {           // Do entire row
               fx = ix ;                             // Convert to fpt
               p = vf * (fx * cosine + fy * sine) ;  // Scaled dist along direc
               dsq = dsqmult * (fx * fx + fy * fy) ; // Distance from center
               weight = vgc * exp ( -0.5 * dsq ) ;   // Window
               *cptr++ = (cos ( p * kparam ) - con) * weight ; // Real part
               *cptr++ = sin ( p * kparam ) * weight ; // And imaginary part
               } // For ix
            } // For iy
         } // For idir
      } // For iv
}

Morlet2D::~Morlet2D ()
{
   if (coefs != NULL)
      free ( coefs ) ;
}

/*
--------------------------------------------------------------------------------

   Transform routine here

--------------------------------------------------------------------------------
*/

void Morlet2D::transform (
   int nrows ,   // Number of rows in image
   int ncols ,   // And columns
   double *img , // Image
   int rcent ,   // Row (origin 0) where filter centered
   int ccent ,   // And column
   double *rt ,  // Real output
   double *it    // Imaginary output
   )
{
   int iscale, ivoice, idir, row, col, decim, roff, coff, maxoff, off ;
   int offspace ;
   double *cptr, *imgptr_up, *imgptr_dn, rsum, isum, scafac, *cent ;

/*
   Make sure that we do not overrun the image with the filter.
   Recompute img to point to the center pixel in the filtered span.
*/

   if ((rcent < bord)  ||  (rcent >= nrows-bord))
      return ;
   if ((ccent < bord)  ||  (ccent >= ncols-bord))
      return ;

   img += rcent * ncols + ccent ; // Point to user's center

/*
   Apply the filters.  Remember that an element in the bottom half of the
   filter is the complex conjugate of the element reflected along both axes.
   Therefore we can implement the full filter very easily by summing left-
   to-right for the top half and right-to-left (flipping the sine sign) for
   the bottom half.
*/

   decim = 1 ;               // Filter coefs decimated for smaller scales
   scafac = 1.0 ;            // Scaling factor
   maxoff = nsmall * space ; // Max offset of centers
   offspace = 2 * maxoff ;   // Space between offset centers

   for (iscale=0 ; iscale<ns ; iscale++) { // Start with longest filter
      off = iscale  ?  maxoff : 0 ;    // Largest scale has only one center
      for (roff=-off ; roff<=off ; roff+=offspace) {     // Centers in square
         for (coff=-off ; coff<=off ; coff+=offspace) {  // around user's cent
            cptr = coefs ;                               // All coefs here
            cent = img + roff * ncols + coff ;           // Offset center
            for (ivoice=0 ; ivoice<nv ; ivoice++) {      // Do all voices
               for (idir=0 ; idir<nd ; idir++) {         // Do all orientations
                  rsum = isum = 0.0 ;                    // Will cumulate here
                  for (row=0 ; row<=hl/decim ; row++) {  // Only saved top half coefs
                     imgptr_up = cent - row * ncols - hl / decim ; // Rows above
                     imgptr_dn = cent + row * ncols + hl / decim ; // Below
                     if (row) {                        // Do both rows at once
                        for (col=-hl ; col<=hl ; col+=decim) {// All columns
                           rsum += *cptr * *imgptr_up ;       // Row above cent
                           rsum += *cptr * *imgptr_dn ;       // And below
                           isum += *(cptr+1) * *imgptr_up++ ; // Imaginary part
                           isum -= *(cptr+1) * *imgptr_dn-- ; // Sign flips
                           cptr += 2 * decim ;                // Skip real, imag
                           }
                        }
                     else {                              // Do center row alone
                        for (col=-hl ; col<=hl ; col+=decim) {// All columns
                           rsum += *cptr * *imgptr_up ;
                           isum += *(cptr+1) * *imgptr_up++ ;
                           cptr += 2 * decim ;           // Skip real, imag
                           } // For all cols
                        } // Else center row
                     cptr -= 2 * (decim-1) ;             // May skip too many
                     if (row != hl/decim)
                        cptr += 2 * (2 * hl + 1) * (decim-1) ; // Decim rows
                     } // For all rows

                  *rt++ = scafac * rsum ;  // Output for this voice and dir
                  *it++ = scafac * isum ;
                  } // For all directions
               } // For all voices
            } // For all column center offsets
         } // For all row center offsets
      decim *= 2 ;    // Decimate coefs for smaller scales
      scafac *= 2.0 ; // Scaling factor
      offspace /= 2 ; // Space between offset centers
      } // For all scales
}
