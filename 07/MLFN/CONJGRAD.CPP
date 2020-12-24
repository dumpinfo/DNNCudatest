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
/*  conjgrad - Conjugate gradient learning                                    */
/*                                                                            */
/*  Normally this returns the scaled mean square error.                       */
/*  If the user interrupted, it returns the negative mean square error.       */
/*  Insufficient memory returns -1.e30.                                       */
/*                                                                            */
/*  This routine contains much debugging code that can be activated by        */
/*  setting the value of DEBUG.  One of the most useful debugging facilities  */
/*  is that the computed gradient can be verified numerically.  This is       */
/*  immensely valuable to those who write their own error functions.          */
/*  But be warned that numerical verification is SLOW.                        */
/*                                                                            */
/******************************************************************************/

#define DEBUG 0

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

double LayerNet::conjgrad (
   TrainingSet *tptr , // Training set to use
   int maxits ,        // Maximum iterations allowed, 0 if no limit
   double reltol ,     // Relative error change tolerance
   double errtol ,     // Quit if error drops this low
   int progress        // Report progress to screen?
   )
{
   int i, n, iter, key, retry, max_retry, bad_cnt ;
   double gam, *g, *h, *work1, *work2, *grad, *base ;
   double error, maxgrad ;
   double prev_err, improvement ;
   int prog_cnt=0 ;
                     
   max_retry = 5 ;

/*
   Allocate work memory.
   Work1 is used for hidden layer 2 deltas in REAL model, and output
   activation partial derivatives and deltas in all COMPLEX models.
   Work2 is output deltas in REAL model, error difference in COMPLEX models.
*/

   MEMTEXT ( "CONJGRAD work" ) ;
   if (nhid2)       // Must be REAL model if this is true
      n = nhid2 ;
   else if (model == NETMOD_COMPLEX_INPUT)
      n = nhid1  ?  nout * 2 + nhid1 * 2  :  nout * 2 ;
   else if (model == NETMOD_COMPLEX_HIDDEN)
      n = nout * 4  +  nhid1 * 4 ;
   else if (model == NETMOD_COMPLEX)
      n = nhid1  ?  nout * 6  +  nhid1 * 4  :  nout * 4 ;
   else
      n = 0 ;

   if (n) {
      work1 = (double *) MALLOC ( n * sizeof(double) ) ;
      if (work1 == NULL)
         return -1.e30 ;
      }
   else
      work1 = NULL ;

   work2 = (double *) MALLOC ( nout_n * sizeof(double) ) ;

   if (nhid1 == 0)               // No hidden layer
      n = nout * nin_n ;
   else if (nhid2 == 0)          // One hidden layer
      n = nhid1 * nin_n + nout * nhid1_n ;
   else                          // Two hidden layers
      n = nhid1 * nin_n + nhid2 * nhid1_n + nout * nhid2_n ;

   grad = (double *) MALLOC ( n * sizeof(double) ) ;
   base = (double *) MALLOC ( n * sizeof(double) ) ;
   g = (double *) MALLOC ( n * sizeof(double) ) ;
   h = (double *) MALLOC ( n * sizeof(double) ) ;

   if ((work2 == NULL) || (grad == NULL) ||
       (base == NULL) || (g == NULL) || (h == NULL)) {
      if (work1 != NULL)
         FREE ( work1 ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      if (grad != NULL)
         FREE ( grad ) ;
      if (base != NULL)
         FREE ( base ) ;
      if (g != NULL)
         FREE ( g ) ;
      if (h != NULL)
         FREE ( h ) ;
      return -1.e30 ;   // Flags error
      }

   prev_err = 1.e30 ;
   error = gradient ( tptr , work1 , work2 , grad ) ;

#if DEBUG > 1
   printf ( "\nCJ starting error = %lf (%lf)", error, trial_error ( tptr ) ) ;
   check_grad ( tptr , grad ) ;
#endif

   memcpy ( g , grad , n * sizeof(double) ) ;
   memcpy ( h , grad , n * sizeof(double) ) ;

   bad_cnt = 0 ;  // Zero gamma to use gradient if CJ failing

/*
   Main iteration loop is here
*/

   iter = 0 ;
   for (;;) {  // Each iter is an epoch

      if ((maxits > 0)  &&  (iter++ >= maxits))
         break ;

/*
   Check current error against user's max.  Abort if user pressed ESCape
*/

      if (error <= errtol)   // If our error is within user's limit
         break ;             // then we are done!

      if (error <= reltol)   // Generally not necessary: reltol<errtol in
         break ;             // practice, but help silly users

      if (kbhit()) {         // Was a key pressed?
         key = getch () ;    // Read it if so
         while (kbhit())     // Flush key buffer in case function key
            getch () ;       // or key was held down
         if (key == 27) {    // ESCape
            error = -error ; // Flags user that ESCape was pressed
            break ;
            }
         }

/*
   Normalize gradient to reasonable values before line minimization
*/

      maxgrad = 1.e-30 ;  // Avoid dividing by zero later
      for (i=0 ; i<n ; i++)
         if (fabs(grad[i]) > maxgrad)
            maxgrad = fabs(grad[i]) ;
      for (i=0 ; i<n ; i++)
         grad[i] /= maxgrad ;

      prev_err = error ;
      error = direcmin ( tptr , error , 30 , 1.e-13 ,
                         1.e-3 , base , grad ) ;
      if (fabs(error) > prev_err+1.e-10)
         printf ( "\n\a\a\aInternal ERROR 1 in CONJGRAD: %.9lf %.9lf",
                  prev_err, error ) ;
      if (error < 0.0)  // Indicates user pressed ESCape
         goto CGFINISH ;

#if DEBUG
      printf ( "\nDIRECMIN improvement = %lf %%",
               100. * (prev_err - error) / prev_err ) ;
#endif

      if ((2.0 * (prev_err - error)) <=       // If this direc gave poor result
          (reltol * (prev_err + error + 1.e-13))) { // will use random direc
         prev_err = error ;
         for (retry=0 ; retry<max_retry ; retry++) {
#if DEBUG
            printf ( "\nRETRY %d with random, last err=%lf", retry, error ) ;
#endif
            for (i=0 ; i<n ; i++)
               grad[i] = (double) (rand() - RANDMAX/2) / (double) RANDMAX ;
            error = direcmin ( tptr , error , 30 , 1.e-13 ,
                               1.e-3 , base , grad ) ;
            if (error < 0.0)  // Indicates user pressed ESCape
               goto CGFINISH ;
            } // For retry
         error = gradient ( tptr , work1 , work2 , grad ) ; // One last grad try
         maxgrad = 1.e-30 ;  // Avoid dividing by zero later
         for (i=0 ; i<n ; i++)
            if (fabs(grad[i]) > maxgrad)
               maxgrad = fabs(grad[i]) ;
         for (i=0 ; i<n ; i++)
            grad[i] /= maxgrad ;
         error = direcmin ( tptr , error , 40 , 1.e-13 ,
                            1.e-6 , base , grad ) ;
         if (error < 0.0)  // Indicates user pressed ESCape
            goto CGFINISH ;
         if ((2.0 * (prev_err - error)) <=           // If random dirs and grad
             (reltol * (prev_err + error + 1.e-13))) // didn't do it
            break ;                                  // then give up
         memcpy ( g , grad , n * sizeof(double) ) ;
         memcpy ( h , grad , n * sizeof(double) ) ;
#if DEBUG
         printf ( "\nSUCCESSFUL RETRY (%d) improvement=%.4lf%%", retry,
               100. * (prev_err - error) / prev_err ) ;
#endif
         } // If this dir gave poor result

      improvement = (prev_err - error) / prev_err ;
      prev_err = error ;

/*
   Setup for next iteration
*/

      error = gradient ( tptr , work1 , work2 , grad ) ;
      if (fabs(error) > prev_err+1.e-10) {
         printf ( "\n\a\a\aInternal ERROR 4 in CONJGRAD: %.10lf %.10lf",
                  prev_err, error ) ;
         getch () ;
         }
      gam = gamma ( g , grad ) ;
#if DEBUG > 1
      printf ( "\nGamma = %lf", gam ) ;
      check_grad ( tptr , grad ) ;
#endif
      if (gam < 0.0)
         gam = 0.0 ;

      if (gam > 5.0)          // limit gamma
         gam = 5.0 ;

/*
   Count how many times in a row we got poor improvement.
   If a few times, do not let gamma exceed one.
   If many times, set gamma=0 to force use of gradient.
*/

      if (improvement < 0.0001)  // Count how many times we
         ++bad_cnt ;             // got tiny improvement
      else                       // in a row
         bad_cnt = 0 ;

      if (bad_cnt >= 2) {        // If a few times
         if (gam > 1.0)          // limit gamma
            gam = 1.0 ;
         }

      if (bad_cnt >= 6) {        // If too many times
         bad_cnt = 0 ;           // set gamma to 0
         gam = 0.0 ;             // to use gradient
#if DEBUG > 1
         printf ( "\nSetting Gamma=0" ) ;
#endif
         }

      find_new_dir ( gam , g , h , grad ) ;

/*
   Diagnostic code
*/

#if DEBUG
      printf ( "\n\nError = %lf", error ) ;
#if DEBUG > 2
      printf ( "grad: " ) ;
      for (i=0 ; i<n ; i++)
         printf ( " %7.3lf", grad[i] ) ;
#endif
#endif

      if (progress  &&  ++prog_cnt == 1000) {
         prog_cnt = 0 ;
         printf ( " (%lf)", error ) ;
         }
      }  // This is the end of the main iteration loop

#if DEBUG
   printf ( "\n\aDone=%lf  Press space...", error ) ;
   while (kbhit())
      getch() ;
   getch() ;
#endif

/*
   Free work memory
*/

CGFINISH:
   MEMTEXT ( "CONJGRAD work" ) ;
   if (work1 != NULL)
      FREE ( work1 ) ;
   FREE ( work2 ) ;
   FREE ( grad ) ;
   FREE ( base ) ;
   FREE ( g ) ;
   FREE ( h ) ;

#if DEBUG > 1
   printf ( "\nCONJGRAD returning %lf", error ) ;
#endif
   return error ;
}

/*
--------------------------------------------------------------------------------

   Local routine to find gamma

--------------------------------------------------------------------------------
*/

double LayerNet::gamma ( double *g , double *grad )
{
   int i, n ;
   double denom, numer ;

   if (nhid1 == 0)        // No hidden layer
      n = nout * nin_n ;
   else if (nhid2 == 0)   // One hidden layer
      n = nhid1 * nin_n + nout * nhid1_n ;
   else                   // Two hidden layers
      n = nhid1 * nin_n + nhid2 * nhid1_n + nout * nhid2_n ;

   numer = denom = 0. ;

   for (i=0 ; i<n ; i++) {
      denom += g[i] * g[i] ;
      numer += (grad[i] - g[i]) * grad[i] ;  // Grad is neg gradient
      }

   if (denom == 0.)   // Should never happen (means gradient is zero!)
      return 0. ;
   else
      return numer / denom ;
}

/*
--------------------------------------------------------------------------------

   Local routine to find correction for next iteration

--------------------------------------------------------------------------------
*/

void LayerNet::find_new_dir ( double gam , double *g ,
                              double *h , double *grad )
{
   int i, n ;

   if (nhid1 == 0)        // No hidden layer
      n = nout * nin_n ;
   else if (nhid2 == 0)   // One hidden layer
      n = nhid1 * nin_n + nout * nhid1_n ;
   else                   // Two hidden layers
      n = nhid1 * nin_n + nhid2 * nhid1_n + nout * nhid2_n ;

   for (i=0 ; i<n ; i++) {
      g[i] = grad[i] ;
      grad[i] = h[i] = g[i] + gam * h[i] ;
      }
}


/*
--------------------------------------------------------------------------------

   Local routine for debugging

--------------------------------------------------------------------------------
*/

#define DELTA 0.000001
void LayerNet::check_grad ( TrainingSet *tptr , double *grad )
{
   int i, j, n ;
   double f0, f1, deriv, dot, len1, len2 ;

   dot = len1 = len2 = 0.0 ;

#if DEBUG > 2
   printf ( "\nHID1: " ) ;
#endif
   for (i=0 ; i<nhid1 ; i++) {
      for (j=0 ; j<nin_n ; j++) {
         hid1_coefs[i*nin_n+j] += DELTA ;
         f0 = trial_error ( tptr ) ;
         hid1_coefs[i*nin_n+j] -= 2.0 * DELTA ;
         f1 = trial_error ( tptr ) ;
         hid1_coefs[i*nin_n+j] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
#if DEBUG > 2
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * *grad ) ;
#endif
         len1 += *grad * *grad ;
         len2 += deriv * deriv ;
         dot += *grad++ * deriv ;
         }
      }

#if DEBUG > 2
   printf ( "\nHID2: " ) ;
#endif
   for (i=0 ; i<nhid2 ; i++) {
      for (j=0 ; j<nhid1_n ; j++) {
         hid2_coefs[i*nhid1_n+j] += DELTA ;
         f0 = trial_error ( tptr ) ;
         hid2_coefs[i*nhid1_n+j] -= 2.0 * DELTA ;
         f1 = trial_error ( tptr ) ;
         hid2_coefs[i*nhid1_n+j] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
#if DEBUG > 2
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * *grad ) ;
#endif
         len1 += *grad * *grad ;
         len2 += deriv * deriv ;
         dot += *grad++ * deriv ;
         }
      }

#if DEBUG > 2
   printf ( "\nOUT: " ) ;
#endif
   if (nhid1 == 0)        // No hidden layer
      n = nin_n ;
   else if (nhid2 == 0)   // One hidden layer
      n = nhid1_n ;
   else                   // Two hidden layers
      n = nhid2_n ;
   for (i=0 ; i<nout ; i++) {
      for (j=0 ; j<n ; j++) {
         out_coefs[i*n+j] += DELTA ;
         f0 = trial_error ( tptr ) ;
         out_coefs[i*n+j] -= 2.0 * DELTA ;
         f1 = trial_error ( tptr ) ;
         out_coefs[i*n+j] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
#if DEBUG > 2
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * *grad ) ;
#endif
         len1 += *grad * *grad ;
         len2 += deriv * deriv ;
         dot += *grad++ * deriv ;
         }
      }
#if DEBUG > 1
   printf ( "\nDOT=%lf", dot / sqrt(len1 * len2) ) ;
#endif
}
