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
#include <string.h>

void qsort_i ( int first , int last , int array[] ) ;

class ToneText {

public:
   ToneText ( int rows , int cols , double taper , int *ok ) ;
   ~ToneText () ;
   void getpix ( int nrows , int ncols , unsigned char *img ,
                 int rcent , int ccent ) ;
   double mean () ;
   double mean_dev () ;
   double std_dev () ;
   double skewness () ;
   double ncon1 () ;
   double ncon2 () ;
   double rel_tone () ;
   double hgrad () ;
   double vgrad () ;
   double median () ;
   double range () ;
   double iq_range () ;
   double skew10 () ;

private:
   int nr ;             // Row radius of area
   int nc ;             // Column radius of area
   int ntot ;           // Number of pixels in area
   float *window ;      // Weight window stored here
   double wt_tot ;      // Total of all weights
   double mean_rad ;    // Mean radius of area
   int *unsorted ;      // Raw pixels from image put here
   int *sorted ;        // Sorted here if needed
   int sort_done ;      // Have they been sorted?
   double p_mean ;      // Mean tone in array, -1.0 if not done
   double p_stddev ;    // Std dev tone in array, -1.0 if not done

   void sort () ;       // Local routine to sort data
} ;

/*
--------------------------------------------------------------------------------

   Constructor, destructor

--------------------------------------------------------------------------------
*/

ToneText::ToneText (
   int rows ,         // Row radius (above and below center)
   int cols ,         // Column radius (left and right of center)
   double taper ,     // Fraction of max at (row, 0), (0, col) etc.
   int *ok            // Memory allocation go ok?
   )
{
   int r, c ;
   float *wptr ;
   double dr, dc, winfac, wt ;

   window = NULL ;
   unsorted = NULL ;
   sorted = NULL ;
   *ok = 0 ;

   nr = rows ;
   nc = cols ;

   ntot = (2 * nr + 1) * (2 * nc + 1) ;

/*
   Allocate memory
*/

   if (taper <= 0.99999999) {  // No window if flat
      if ((window = (float *) malloc ( ntot * sizeof(float) )) == NULL)
         return ;
      }

   unsorted = (int *) malloc ( ntot * sizeof(int) ) ;
   if (unsorted != NULL)
      sorted = (int *) malloc ( ntot * sizeof(int) ) ;

   if (sorted == NULL) {
      if (window != NULL) {
         free ( window ) ;
         window = NULL ;  // So destructor doesn't try to free it
         }
      if (unsorted != NULL) {
         free ( unsorted ) ;
         unsorted = NULL ;
         }
      return ;
      }

   *ok = 1 ;   // Signal caller that memory allocs ok

/*
   Compute the mean radius of the area.  This is needed by rel_tone.
*/

   mean_rad = 0.0 ;
   for (r=-nr ; r<=nr ; r++) {
      for (c=-nc ; c<=nc ; c++)
         mean_rad += sqrt ( r * r + c * c ) ;
      }
   mean_rad /= ntot ;

/*
   Compute window if needed.  No window if flat taper.
*/

   if (window == NULL)
      return ;

   if (taper <= 0.0)    // Protection from silly user
      taper = 1.e-30 ;

   winfac = log ( taper ) ;
   wptr = window ;
   wt_tot = 0.0 ;
   for (r=-nr ; r<=nr ; r++) {
      dr = (double) r / (double) nr ;
      dr = dr * dr ;
      for (c=-nc ; c<=nc ; c++) {
         dc = (double) c / (double) nc ;
         dc = dc * dc ;
         wt = exp ( winfac * (dr + dc) ) ; // Weight for this position
         *wptr++ = wt ;                    // Save it
         wt_tot += wt ;                    // And cumulate sum
         }
      }
}

ToneText::~ToneText ()
{
   if (window != NULL)
      free ( window ) ;
   if (unsorted != NULL)
      free ( unsorted ) ;
   if (sorted != NULL)
      free ( sorted ) ;
}

/*
--------------------------------------------------------------------------------

   getpix - Transfer the pixels from the image to local array

--------------------------------------------------------------------------------
*/

void ToneText::getpix (
   int nrows ,          // Number of rows in image
   int ncols ,          // And columns
   unsigned char *img , // Image
   int rcent ,          // Row (origin 0) where area centered
   int ccent            // And column
   )
{
   int r, c, *uptr ;
   unsigned char *iptr ;

/*
   Make sure we do not overrun the image.
*/

   if ((rcent < nr)  ||  (rcent >= nrows-nr))
      return ;
   if ((ccent < nc)  ||  (ccent >= ncols-nc))
      return ;

   uptr = unsorted ;
   for (r=rcent-nr ; r<=rcent+nr ; r++) {
      iptr = img + r * ncols + ccent - nc ;
      c = 2 * nc + 1 ;
      while (c--)
         *uptr++ = *iptr++ ;
      }

/*
   Flag some local variables as not being found yet
*/

   sort_done = 0 ;
   p_mean = p_stddev = -1.0 ;
}

/*
   Simple tone-texture variables
*/

double ToneText::mean ()    //---------> Mean of the area
{
   int i, *uptr ;
   float *wptr ;

   if (p_mean >= 0.0)       // Already computed?
      return p_mean ;

   p_mean = 0.0 ;           // Will cumulate and save mean here
   i = ntot ;               // This many pixels
   uptr = unsorted ;        // They are here

   if (window == NULL) {    // If no window just sum
      while (i--)
         p_mean += *uptr++ ;
      p_mean /= (double) ntot ;
      }

   else {
      wptr = window ;       // Window weights
      while (i--)
         p_mean += *uptr++  *  *wptr++ ;
      p_mean /= wt_tot ;
      }

   return p_mean ;
}

double ToneText::mean_dev ()    //---------> Mean deviation of the area
{
   int i, *uptr ;
   float *wptr ;
   double mdev ;

   mdev = mean () ;         // Need mean, mdev just to silence warnings

   mdev = 0.0 ;             // Will cumulate and save mean_dev here
   i = ntot ;               // This many pixels
   uptr = unsorted ;        // They are here

   if (window == NULL) {    // If no window just sum
      while (i--)
         mdev += fabs ( (double) *uptr++  -  p_mean ) ;
      mdev /= (double) ntot ;
      }

   else {
      wptr = window ;       // Window weights
      while (i--)
         mdev += fabs ( (double) *uptr++  -  p_mean )  *  *wptr++ ;
      mdev /= wt_tot ;
      }

   return mdev ;
}



double ToneText::std_dev ()    //---------> Standard deviation of the area
{
   int i, *uptr ;
   float *wptr ;
   double diff ;

   if (p_stddev >= 0.0)    // Already computed?
      return p_stddev ;

   diff = mean () ;        // Need mean, diff just to silence warnings

   p_stddev = 0.0 ;        // Will cumulate and save std_dev here
   i = ntot ;              // This many pixels
   uptr = unsorted ;       // They are here

   if (window == NULL) {   // If no window just sum
      while (i--) {
         diff = *uptr++  -  p_mean ;
         p_stddev += diff * diff ;
         }
      p_stddev = sqrt ( p_stddev / (double) ntot ) ;
      }

   else {
      wptr = window ;      // Window weights
      while (i--) {
         diff = *uptr++  -  p_mean ;
         p_stddev += diff * diff * *wptr++ ;
         }
      p_stddev = sqrt ( p_stddev / wt_tot ) ;
      }

   return p_stddev ;
}


double ToneText::skewness ()    //---------> Skewness of the area
{
   int i, *uptr ;
   float *wptr ;
   double diff, std, skew ;

   std = std_dev () ;      // Need standard deviation

   skew = 0.0 ;            // Will cumulate and save skewness here
   i = ntot ;              // This many pixels
   uptr = unsorted ;       // They are here

   if (window == NULL) {   // If no window just sum
      while (i--) {
         diff = *uptr++  -  p_mean ;
         skew += diff * diff * diff ;
         }
      skew /= (double) ntot ;
      }

   else {
      wptr = window ;      // Window weights
      while (i--) {
         diff = *uptr++  -  p_mean ;
         skew += diff * diff * diff * *wptr++ ;
         }
      skew /= wt_tot ;
      }

   return skew / (std * std * std) ;
}

double ToneText::ncon1 ()    //---------> First neighbor contrast
{
   int r, c, curr, prev, *uptr, nrb, ncb ;
   float *wptr ;
   double con, wt, wtsum, wprev, wcurr ;

   con = 0.0 ;              // Will cumulate and save contrast here
   nrb = 2 * nr + 1 ;       // Number of rows in box
   ncb = 2 * nc + 1 ;       // And columns

   if (window == NULL) {                 // If no window just sum
      for (r=0 ; r<nrb ; r++) {          // First, do left nbrs for all rows
         uptr = unsorted + r * ncb ;     // First pix in this row
         curr = *uptr++ ;                // This is its tone
         c = ncb ;                       // Number of columns in box
         while (--c) {                   // Do one less because pairs
            prev = curr ;                // Pixel to left of current
            curr = *uptr++ ;             // New current pixel
            con += abs ( curr - prev ) ; // Cumulate contrast
            }
         }
      for (c=0 ; c<ncb ; c++) {          // Now, do top nbrs for all cols
         uptr = unsorted + c ;           // First pix in this column
         curr = *uptr ;                  // This is its tone
         r = nrb ;                       // Number of rows in box
         while (--r) {                   // Do one less because pairs
            prev = curr ;                // Pixel above current
            uptr += ncb ;                // Advance down one row
            curr = *uptr ;               // New current pixel
            con += abs ( curr - prev ) ; // Cumulate contrast
            }
         }
      con /= nrb * (ncb-1)  +  ncb * (nrb-1) ; // Number of pairs used
      }

   else {
      wtsum = 0.0 ;                      // Will sum derived weights here
      for (r=0 ; r<nrb ; r++) {          // First, do left nbrs for all rows
         uptr = unsorted + r * ncb ;     // First pix in this row
         wptr = window + r * ncb;        // Ditto window weights
         curr = *uptr++ ;                // This is its tone
         wcurr = *wptr++ ;               // And weight
         c = ncb ;                       // Number of columns in box
         while (--c) {                   // Do one less because pairs
            prev = curr ;                // Pixel to left of current
            wprev = wcurr ;              // Ditto for weights
            curr = *uptr++ ;             // New current pixel
            wcurr = *wptr++ ;            // And weight
            wt = wcurr + wprev ;         // Contrast weight includes both
            con += abs ( curr - prev ) * wt ; // Cumulate contrast
            wtsum += wt ;                // And weights
            }
         }
      for (c=0 ; c<ncb ; c++) {          // Now, do top nbrs for all cols
         uptr = unsorted + c ;           // First pix in this column
         wptr = window + c ;             // Ditto window weights
         curr = *uptr ;                  // This is its tone
         wcurr = *wptr ;                 // And weight
         r = nrb ;                       // Number of rows in box
         while (--r) {                   // Do one less because pairs
            prev = curr ;                // Pixel above current
            wprev = wcurr ;              // Ditto for weights
            uptr += ncb ;                // Advance down one row
            wptr += ncb ;                // Ditto weights
            curr = *uptr ;               // New current pixel
            wcurr = *wptr ;              // And weight
            wt = wcurr + wprev ;         // Contrast weight includes both
            con += abs ( curr - prev ) * wt ; // Cumulate contrast
            wtsum += wt ;                // And weights
            }
         }
      con /= wtsum ;
      }

   return con ;
}


double ToneText::ncon2 ()    //---------> Second neighbor contrast
{
   int r, c, curr, prev, *uptr, nrb, ncb ;
   float *wptr ;
   double con, wt, wtsum, wprev, wcurr ;

   con = 0.0 ;              // Will cumulate and save contrast here
   nrb = 2 * nr + 1 ;       // Number of rows in box
   ncb = 2 * nc + 1 ;       // And columns

   if (window == NULL) {                 // If no window just sum
      for (r=0 ; r<nrb ; r++) {          // First, do left nbrs for all rows
         uptr = unsorted + r * ncb ;     // First pix in this row
         c = ncb - 2 ;                   // Number of pairs in box
         while (c--) {                   // Do all pairs
            curr = *(uptr+2) ;           // One pixel
            prev = *uptr++ ;             // Pixel 2 to left of current
            con += abs ( curr - prev ) ; // Cumulate contrast
            }
         }
      for (c=0 ; c<ncb ; c++) {          // Now, do top nbrs for all cols
         uptr = unsorted + c ;           // First pix in this column
         r = nrb - 2 ;                   // Number of pairs in box
         while (r--) {                   // Do one less because pairs
            prev = *uptr ;               // One pixel
            uptr += ncb ;                // Advance down one row
            curr = *(uptr + ncb) ;       // Pixel 2 rows beyond other
            con += abs ( curr - prev ) ; // Cumulate contrast
            }
         }
      con /= nrb * (ncb-2)  +  ncb * (nrb-2) ; // Number of pairs used
      }

   else {
      wtsum = 0.0 ;                      // Will sum derived weights here
      for (r=0 ; r<nrb ; r++) {          // First, do left nbrs for all rows
         uptr = unsorted + r * ncb ;     // First pix in this row
         wptr = window + r * ncb;        // Ditto window weights
         c = ncb - 2 ;                   // Number of pairs in box
         while (c--) {                   // Do one less because pairs
            curr = *(uptr+2) ;           // One pixel
            prev = *uptr++ ;             // Pixel 2 to left of current
            wcurr = *(wptr+2) ;          // One weight
            wprev = *wptr++ ;            // Weight 2 to left of current
            wt = wcurr + wprev ;         // Contrast weight includes both
            con += abs ( curr - prev ) * wt ; // Cumulate contrast
            wtsum += wt ;                // And weights
            }
         }
      for (c=0 ; c<ncb ; c++) {          // Now, do top nbrs for all cols
         uptr = unsorted + c ;           // First pix in this column
         wptr = window + c ;             // Ditto window weights
         r = nrb - 2 ;                   // Number of pairs in box
         while (r--) {                   // Do one less because pairs
            prev = *uptr ;               // One pixel
            uptr += ncb ;                // Advance down one row
            curr = *(uptr + ncb) ;       // Pixel 2 rows beyond other
            wprev = *wptr ;              // One pixel
            wptr += ncb ;                // Advance down one row
            wcurr = *(wptr + ncb) ;      // Pixel 2 rows beyond other
            wt = wcurr + wprev ;         // Contrast weight includes both
            con += abs ( curr - prev ) * wt ; // Cumulate contrast
            wtsum += wt ;                // And weights
            }
         }
      con /= wtsum ;
      }

   return con ;
}


double ToneText::rel_tone ()    //---------> Relative tone of center
{
   int r, c, *uptr ;
   float *wptr ;
   double sxy, ssx, ymean, x ;

   ymean = mean () ;       // Mean tone

   sxy = 0.0 ;             // Will cumulate cross-product here
   ssx = 0.0 ;             // Sum of squares of distance (x)
   uptr = unsorted ;       // Pixel tones are here

   if (window == NULL) {    // If no window just sum
      for (r=-nr ; r<=nr ; r++) {
         for (c=-nc ; c<=nc ; c++) {
            x = mean_rad - sqrt ( r * r + c * c ) ;
            sxy += x  *  ((double) *uptr++  -  ymean ) ;
            ssx += x * x ;
            }
         }
      }

   else {
      wptr = window ;       // Window weights
      for (r=-nr ; r<=nr ; r++) {
         for (c=-nc ; c<=nc ; c++) {
            x = mean_rad - sqrt ( r * r + c * c ) ;
            sxy += x  *  ((double) *uptr++  -  ymean )  *  *wptr ;
            ssx += x * x * *wptr++ ;
            }
         }
      }

   return sxy / ssx ;
}


double ToneText::hgrad ()    //---------> Horizontal tone gradient
{
   int r, c, *uptr ;
   float *wptr ;
   double sxy, ssx, ymean ;

   ymean = mean () ;       // Mean tone

   sxy = 0.0 ;             // Will cumulate cross-product here
   ssx = 0.0 ;             // Sum of squares of distance (x)
   uptr = unsorted ;       // Pixel tones are here

   if (window == NULL) {    // If no window just sum
      for (r=-nr ; r<=nr ; r++) {
         for (c=-nc ; c<=nc ; c++) {
            sxy += c  *  ((double) *uptr++  -  ymean ) ;
            ssx += c * c ;
            }
         }
      }

   else {
      wptr = window ;       // Window weights
      for (r=-nr ; r<=nr ; r++) {
         for (c=-nc ; c<=nc ; c++) {
            sxy += c  *  ((double) *uptr++  -  ymean )  *  *wptr ;
            ssx += c * c * *wptr++ ;
            }
         }
      }

   return sxy / ssx ;
}


double ToneText::vgrad ()    //---------> Vertical tone gradient
{
   int r, c, *uptr ;
   float *wptr ;
   double sxy, ssx, ymean ;

   ymean = mean () ;       // Mean tone

   sxy = 0.0 ;             // Will cumulate cross-product here
   ssx = 0.0 ;             // Sum of squares of distance (x)
   uptr = unsorted ;       // Pixel tones are here

   if (window == NULL) {    // If no window just sum
      for (r=-nr ; r<=nr ; r++) {
         for (c=-nc ; c<=nc ; c++) {
            sxy += r  *  ((double) *uptr++  -  ymean ) ;
            ssx += r * r ;
            }
         }
      }

   else {
      wptr = window ;       // Window weights
      for (r=-nr ; r<=nr ; r++) {
         for (c=-nc ; c<=nc ; c++) {
            sxy += r  *  ((double) *uptr++  -  ymean )  *  *wptr ;
            ssx += r * r * *wptr++ ;
            }
         }
      }

   return sxy / ssx ;
}


void ToneText::sort ()    //---------> Local routine to sort data
{
   if (sort_done)
      return ;

   memcpy ( sorted , unsorted , ntot * sizeof(int) ) ;
   qsort_i ( 0 , ntot-1 , sorted ) ;
   sort_done = 1 ;
}


double ToneText::median ()    //---------> Median tone
{
   sort () ;
   return sorted[ntot/2] ;
}


double ToneText::range ()    //---------> Range
{
   sort () ;
   return sorted[ntot-1] - sorted[0] ;
}


double ToneText::iq_range ()    //---------> Inter-quartile range
{
   sort () ;
   return sorted[3*ntot/4] - sorted[ntot/4] ;
}


double ToneText::skew10 ()    //---------> Skewness discarding outer 10
{
   int numer, denom, median ;

   sort () ;
   median = sorted[ntot/2] ;
   numer = sorted[9*ntot/10] - median ;
   denom = median - sorted[ntot/10] ;

   if (numer) {
      if (denom)
         return log ( (double) numer / (double) denom ) ;
      else
         return 1.0 ;
      }
   else {
      if (denom)
         return -1.0 ;
      else
         return 0.0 ;
      }
}
