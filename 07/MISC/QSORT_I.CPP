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

/****************************************************************************/
/*                                                                          */
/*  QSORT_I - Quick sort an integer array.                                  */
/*                                                                          */
/****************************************************************************/

void qsort_i (
   int first ,   // Subscript of first element of array to be sorted
   int last ,    // And the last
   int array[]   // Input/output array
   )
{
   int temp, split ;
   register int low, high ;

   split = array[(first+last)/2] ;
   low = first ;
   high = last ;

   do {
      while ( split > array[low] )
         ++low ;
      while ( split < array[high] )
         --high ;
      if (low == high) {
         ++low ;
         --high ;
         }
      else if (low < high) {
         temp = array[low] ;
         array[low++] = array[high] ;
         array[high--] = temp ;
         }
      } while ( low <= high ) ;

   if (low < last)
      qsort_i ( low , last , array ) ;
   if (first < high)
      qsort_i ( first , high , array ) ;
}
