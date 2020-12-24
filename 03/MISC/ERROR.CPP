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

#include <string.h>
#include <math.h>
#include <stdlib.h>

double unifrand () ;

class DataClass {  // Dummy definition; use your own
public:
   double data ;
} ;

/*
--------------------------------------------------------------------------------

   cv - Compute error via cross validation

--------------------------------------------------------------------------------
*/

void cv (
   int n ,            // Number of data points
   DataClass *data ,  // The data is here
   double *app ,      // Apparent error from testing tset
   double *excess     // Excess error (add to app to get pop)
   )
{
   int exclude, first ;
   double errsum ;
   DataClass temp ;

   errsum = 0.0 ;
   exclude = n ;
   first = 1 ;

   while (exclude--) {

      if (first)
         first = 0 ;
      else {               // If not first trial, swap excluded to end
         temp = data[exclude] ;
         data[exclude] = data[n-1] ;
         data[n-1] = temp ;
         }

      train ( n-1 , data ) ;             // Train on first n-1 cases
      errsum += test ( 1 , data+n-1 ) ;  // Test last (excluded) case
      }

   errsum /= n ;                         // Population error

   train ( n , data ) ;                  // Apparent error
   *app = test ( n , data ) ;            // Is needed to compute excess
   *excess = errsum - *app ;
}


/*
--------------------------------------------------------------------------------

   boot - Compute the error via bootstrap

--------------------------------------------------------------------------------
*/

void boot (
   int n ,            // Number of data points
   DataClass *data ,  // The data is here
   double *app ,      // Apparent error from testing tset
   double *excess     // Excess error (add to app to get pop)
   )
{
   int i, rep, sub, *count ;
   int m = 200 ;
   double err, errsum ;
   DataClass *x ;

   if (m < n)      // If the dataset is large
      m = n ;      // Do enough reps to be thorough

   x = (DataClass *) malloc ( n * sizeof(DataClass) ) ; // Bootstraps here
   count = (int *) malloc ( n * sizeof(int) ) ;      // Count uses in bootstrap

   errsum = 0.0 ;

   for (rep=0 ; rep<m ; rep++) {

      memset ( count , 0 , n * sizeof(int) ) ; // Zero usage counter
      for (i=0 ; i<n ; i++) {                  // Bootstrap sample same size
         sub = unifrand() * n ;                // Select this case
         if (sub >= n)                         // Cheap insurance in case
            sub = n-1 ;                        // unifrand() returns 1
         x[i] = data[sub] ;                    // Get this case
         ++count[sub] ;                        // Count its use
         }

      train ( n , x ) ;                        // Train on bootstrap sample
      for (i=0 ; i<n ; i++) {                  // Test all cases
         err = test ( 1 , data+i ) ;           // Error of this case
         errsum += err * (1 - count[i]) ;      // Bootstrap formula
         }
      }

   errsum /= (double) m  *  (double) n ;       // Grand mean

   train ( n , data ) ;                        // Also return the
   *app = test ( n , data ) ;                  // Apparent error
   *excess = errsum ;

   free ( x ) ;
   free ( count ) ;
}


/*
--------------------------------------------------------------------------------

   E0 - Compute the error via E0 method

--------------------------------------------------------------------------------
*/

void E0 (
   int n ,            // Number of data points
   DataClass *data ,  // The data is here
   double *app ,      // Apparent error from testing tset
   double *excess     // Excess error (add to app to get pop)
   )
{
   int i, rep, sub, ntot, *count ;
   int m = 200 ;
   double errsum ;
   DataClass *x ;

   if (m < n)
      m = n ;

   x = (DataClass *) malloc ( n * sizeof(DataClass) ) ;  // Bootstraps here
   count = (int *) malloc ( n * sizeof(int) ) ;       // Count uses in bootstrap

   errsum = 0.0 ;
   ntot = 0 ;

   for (rep=0 ; rep<m ; rep++) {

      memset ( count , 0 , n * sizeof(int) ) ; // Zero usage counter
      for (i=0 ; i<n ; i++) {                  // Bootstrap sample same size
         sub = unifrand() * n ;                // Select this case
         if (sub >= n)                         // Cheap insurance in case
            sub = n-1 ;                        // unifrand() returns 1
         x[i] = data[sub] ;                    // Get this case
         ++count[sub] ;                        // Count its use
         }

      train ( n , x ) ;                        // Train on bootstrap sample

      for (i=0 ; i<n ; i++) {                  // Check all cases
         if (! count[i]) {                     // If not used in training
            errsum += test ( 1 , data+i ) ;    // Find its error
            ++ntot ;                           // Grand test count
            }
         }
      }

   errsum /= ntot ;                            // Mean of all tests

   train ( n , data ) ;                        // Also need the
   *app = test ( n , data ) ;                  // Apparent error
   *excess = errsum - *app ;

   free ( x ) ;
   free ( count ) ;
}
