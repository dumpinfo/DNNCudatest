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

int getch () ;

extern void qsort ( int i , int j , double *data ) ;

double cube ( double x )
{
   return x * x * x ;
}

double cube_root ( double x )
{
   if (x == 0.0)
      return 0.0 ;
   else if (x < 0.0)
      return -pow ( -x , 1.0 / 3.0 ) ;
   else 
      return pow ( x , 1.0 / 3.0 ) ;
}

class Scaler {

public:
   Scaler ( int n , double *r_model , double *i_model , int transform ) ;
   void scale ( int n , double *r_input , double *i_input ,
                double *r_output , double *i_output ) ;
   void unscale ( int n , double *r_input , double *i_input ,
                  double *r_output , double *i_output ) ;

private:
   int is_complex, tfunc ;
   double r_offset, i_offset, factor ;
} ;

/*
   Constructor computes and saves offset and scaling factor
*/

Scaler::Scaler (
   int n ,            // Length of model signal
   double *r_model ,  // Use this signal as model
   double *i_model ,  // It may be complex (This is NULL if real)
   int transform      // Transform also?
   )
{
   int i ;
   double mag, trans, *work, *r_work, *i_work, rtemp, itemp ;

   is_complex = (i_model != NULL) ;
   tfunc = transform ;

   if (n < 2) {   // Make provision for degenerate case
      r_offset = r_model[0] ;
      if (is_complex)
         i_offset = i_model[0] ;
      factor = 1.0 ;
      return ;
      }

   if (is_complex  &&  transform) {
      work = (double *) malloc ( 3 * n * sizeof(double) ) ;
      r_work = work + n ;
      i_work = r_work + n ;
      }
   else {
      work = (double *) malloc ( n * sizeof(double) ) ;
      r_work = r_model ;
      i_work = i_model ;
      }

   if (work == NULL) {
      factor = -1.0 ;    // Flag error
      return ;
      }

   if (is_complex) {

      if (transform) {
         for (i=0 ; i<n ; i++) {
            mag = sqrt ( r_model[i] * r_model[i] + i_model[i] * i_model[i] ) ;
            if (mag > 0.0) {
               trans = cube_root ( mag ) ;
               r_work[i] = r_model[i] * trans / mag ;
               i_work[i] = i_model[i] * trans / mag ;
               }
            else
               r_work[i] = i_work[i] = 0.0 ;
            }
         }

      for (i=0 ; i<n ; i++)
         work[i] = r_work[i] ;
      qsort ( 0 , n-1 , work ) ;
      r_offset = work[n/2] ;

      for (i=0 ; i<n ; i++)
         work[i] = i_work[i] ;
      qsort ( 0 , n-1 , work ) ;
      i_offset = work[n/2] ;

      for (i=0 ; i<n ; i++) {
         rtemp = r_work[i] - r_offset ;
         itemp = i_work[i] - i_offset ;
         work[i] = rtemp * rtemp + itemp * itemp ;
         }
      qsort ( 0 , n-1 , work ) ;

      if (work[n/2] > 0.0)
         factor = 3.0 / sqrt ( work[n/2] ) ;
      else 
         factor = 1.e-30 ;
      }

   else {
      if (transform) {
         for (i=0 ; i<n ; i++)
            work[i] = cube_root ( r_model[i] ) ;
         }
      else {
         for (i=0 ; i<n ; i++)
            work[i] = r_model[i] ;
         }
      qsort ( 0 , n-1 , work ) ;
      r_offset = work[n/2] ;
      factor = 3.0 / (work[3*n/4] - work[n/4]) ;
      }

   free ( work ) ;
}

/*
   Routines for scaling and unscaling
*/

void Scaler::scale ( int n , double *r_input , double *i_input ,
                     double *r_output , double *i_output )
{
   double mag, trans, rtemp, itemp ;

   if (factor < 0.0)
      return ;

   if (is_complex) {

      if (tfunc) {
         while (n--) {
            mag = sqrt ( r_input[n] * r_input[n] + i_input[n] * i_input[n] ) ;
            if (mag > 0.0) {
               trans = cube_root ( mag ) ;
               rtemp = r_input[n] * trans / mag ;
               itemp = i_input[n] * trans / mag ;
               }
            else
               rtemp = itemp = 0.0 ;
            r_output[n] = factor * (rtemp - r_offset) ;
            i_output[n] = factor * (itemp - i_offset) ;
            }
         }

      else {
         while (n--) {
            r_output[n] = factor * (r_input[n] - r_offset) ;
            i_output[n] = factor * (i_input[n] - i_offset) ;
            }
         }
      }
   else {
      if (tfunc) {
         while (n--)
            r_output[n] = factor * (cube_root ( r_input[n] ) - r_offset) ;
         }
      else {
         while (n--)
            r_output[n] = factor * (r_input[n] - r_offset) ;
         }
      }
}

void Scaler::unscale ( int n , double *r_input , double *i_input ,
                       double *r_output , double *i_output )
{
   double fac, mag, trans, rtemp, itemp ;

   if (factor <= 0.0)
      return ;

   fac = 1.0 / factor ;

   if (is_complex) {

      if (tfunc) {
         while (n--) {
            rtemp = r_input[n] * fac + r_offset ;
            itemp = i_input[n] * fac + i_offset ;
            mag = sqrt ( rtemp * rtemp + itemp * itemp ) ;
            if (mag > 0.0) {
               trans = cube ( mag ) ;
               r_output[n] = rtemp * trans / mag ;
               i_output[n] = itemp * trans / mag ;
               }
            else
               r_output[n] = i_output[n] = 0.0 ;
            }
         }

      else {
         while (n--) {
            r_output[n] = r_input[n] * fac + r_offset ;
            i_output[n] = i_input[n] * fac + i_offset ;
            }
         }
      }
   else {
      if (tfunc) {
         while (n--)
            r_output[n] = cube ( r_input[n] * fac + r_offset ) ;
         }
      else {
         while (n--)
            r_output[n] = r_input[n] * fac + r_offset ;
         }
      }
}
