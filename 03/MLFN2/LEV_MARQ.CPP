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
/*  LEV_MARQ - Do Levenberg-Marquardt direct descent learning                 */
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

double LayerNet::lev_marq (
   TrainingSet *tptr , // Training set to use
   int maxits ,        // Maximum iterations allowed, 0 if no limit
   double reltol ,     // Relative error change tolerance
   double errtol ,     // Quit if error drops this low
   int progress        // Report progress to screen?
   )
{
   int i, n, nvars, iter, key, bad_cnt, trivial_cnt, reset_ab ;
   double *work1, *work2, *hessian, *grad, *delta ;
   double error, maxgrad, lambda ;
   double prev_err, improvement ;
   SingularValueDecomp *sptr ;
   int prog_cnt=0 ;
                     
/*
   Allocate work memory.
   Work1 is used for hidden layer 2 deltas in REAL model, and output
   activation partial derivatives and deltas in all COMPLEX models.
   Work2 is for the output activation gradient in all models.
   Hessian and grad are the approximate Hessian and error gradient.
   Delta is the correction vector.
   The total number of variables is nvars.
*/

   if (nhid1 == 0)               // No hidden layer
      nvars = nout * nin_n ;
   else if (nhid2 == 0)          // One hidden layer
      nvars = nhid1 * nin_n + nout * nhid1_n ;
   else                          // Two hidden layers
      nvars = nhid1 * nin_n + nhid2 * nhid1_n + nout * nhid2_n ;

   MEMTEXT ( "LEV_MARQ: new SingularValueDecomp" ) ;
   sptr = new SingularValueDecomp ( nvars , nvars , 0 ) ;
   if ((sptr == NULL)  || ! sptr->ok) {
      if (sptr != NULL)
         delete sptr ;
      return -1.e30 ;
      }

   MEMTEXT ( "LEV_MARQ work" ) ;
   if (nhid2)       // Must be REAL model if this is true
      n = nhid2 ;
   else if (model == NETMOD_COMPLEX_INPUT)
      n = nhid1  ?  nout * 2 + nhid1 * 2  :  nout * 2 ;
   else if (model == NETMOD_COMPLEX_HIDDEN)
      n = nout * 4  +  nhid1 * 4 ;
   else if (model == NETMOD_COMPLEX)
      n = nhid1  ?  nout * 6  +  nhid1 * 4  :  nout * 4 ;
   else
      n = 0 ;   // Real with one hidden layer

   if (n) {
      work1 = (double *) MALLOC ( n * sizeof(double) ) ;
      if (work1 == NULL) {
         delete sptr ;
         return -1.e30 ;
         }
      }
   else
      work1 = NULL ;

   work2 = (double *) MALLOC ( nvars * sizeof(double) ) ;
   hessian = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   grad = (double *) MALLOC ( nvars * sizeof(double) ) ;
   delta = (double *) MALLOC ( nvars * sizeof(double) ) ;

   if ((work2 == NULL) || (hessian == NULL) || (grad == NULL) || (delta == NULL)){
      if (work1 != NULL)
         FREE ( work1 ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      if (hessian != NULL)
         FREE ( hessian ) ;
      if (grad != NULL)
         FREE ( grad ) ;
      if (delta != NULL)
         FREE ( delta ) ;
      delete sptr ;
      return -1.e30 ;   // Flags error
      }

/*
   Compute the error, hessian, and error gradient at the starting point.
*/

   error = lm_core ( tptr , work1 , work2 , hessian , grad ) ;
   prev_err = error ;  // Will be 'previous iteration' error
   reset_ab = 1 ;      // Flag to use most recent good hessian and grad

#if DEBUG > 1
   printf ( "\nLM starting error = %lf (%lf)", error, trial_error ( tptr ) ) ;
   check_lm_grad ( tptr , grad ) ;
#endif

/*
   Every time an iteration results in increased error, increment bad_cnt
   so that remedial action or total escape can be taken.
   Do a similar thing for improvements that are tiny via trivial_cnt.
*/

   bad_cnt = 0 ;       // Counts bad iterations for restart or exit
   trivial_cnt = 0 ;   // Counts trivial improvements for restart or exit

/*
   Initialize lambda to slightly exceed the largest magnitude diagonal
   of the Hessian.
*/

   lambda = 0.0 ;
   for (i=0 ; i<nvars ; i++) {
      if (hessian[i*nvars+i] > lambda)
         lambda = hessian[i*nvars+i] ;
      }

   lambda += 1.e-20 ;

/*
   Main iteration loop is here
*/

   iter = 0 ;
   for (;;) {  // Each iter is an epoch

#if DEBUG
      printf ( "\nLM iter %d  lambda=%lf  err=%lf", iter, lambda, error ) ;
#endif

      if ((maxits > 0)  &&  (iter++ >= maxits))
         break ;

/*
   Check current error against user's max.  Abort if user pressed ESCape
*/

      if (kbhit()) {         // Was a key pressed?
         key = getch () ;    // Read it if so
         while (kbhit())     // Flush key buffer in case function key
            getch () ;       // or key was held down
         if (key == 27) {    // ESCape
            prev_err = -prev_err ; // Flags user that ESCape was pressed
            break ;
            }
         }

      if (error <= errtol)   // If our error is within user's limit
         break ;             // then we are done!

      if (error <= reltol)   // Generally not necessary: reltol<errtol in
         break ;             // practice, but help silly users

      if (reset_ab) {        // Revert to latest good Hessian and gradient?
         memcpy ( sptr->a , hessian , nvars * nvars * sizeof(double) ) ;
         memcpy ( sptr->b , grad , nvars * sizeof(double) ) ;
         }

/*
   Add lambda times the unit diagonal matrix to the Hessian.
   Solve the linear system for the correction, add that correction to the
   current point, and compute the error, Hessian, and gradient there.
*/

      for (i=0 ; i<nvars ; i++)  // Shift diagonal for stability
         sptr->a[i*nvars+i] += lambda ;

      sptr->svdcmp () ;                  // Singular value decomposition
      sptr->backsub ( 1.e-8 , delta ) ;  // Back substitution solves system

      step_out_lm ( 1.0 , delta ) ;      // Jump to new point
      error = lm_core ( tptr , work1 , work2 , sptr->a , sptr->b ) ;

#if DEBUG
      printf ( "  new=%lf", error ) ;
#if DEBUG > 3
      printf ( "\n(Dhess grad): " ) ;
      for (i=0 ; i<nvars ; i++)
         printf ( " (%lf %lf)", sptr->a[i*nvars+i], sptr->b[i] ) ;
#if DEBUG > 4
      check_lm_grad ( tptr , grad ) ;
#endif
#endif
#endif

      improvement = (prev_err - error) / prev_err ;

      if (improvement > 0.0) {
#if DEBUG
         printf ( "   GOOD = %lf%%", 100.0 * improvement ) ;
#endif

/*
   This correction resulted in improvement.  If only a trivial amount,
   check the gradient (relative to the error).  If also small, quit.
   Otherwise count these trivial improvements.  If there were a few,
   the Hessian may be bad, so retreat toward steepest descent.  If there
   were a lot, give up.
*/

         prev_err = error ;           // Keep best error here
         if (improvement < reltol) {
            maxgrad = 0.0 ;
            for (i=0 ; i<nvars ; i++) {
               if (fabs ( sptr->b[i] )  >  maxgrad)
                  maxgrad = fabs ( sptr->b[i] ) ;
               }
            if (error > 1.0)
               maxgrad /= error ;
#if DEBUG
            printf ( "   Triv=%d  mg=%lf", trivial_cnt, maxgrad ) ;
#endif
            if (maxgrad <= reltol)
               break ;

            if (trivial_cnt++ == 4) {
               for (i=0 ; i<nvars ; i++) {
                  if (hessian[i*nvars+i] > lambda)
                     lambda = hessian[i*nvars+i] ;
                  }
               }
            else if (trivial_cnt == 10)  // Normal escape from loop
               break ;
            }
         else
            trivial_cnt = 0 ; // Reset counter whenever good improvement

/*
   Since this step was good, update everything: the Hessian, the gradient,
   and the 'previous iteration' error.  Zero reset_ab so that we do not
   waste time copying the Hessian and gradient into sptr, as they are
   already there.  Cut lambda so that we approach Newton's method.
*/

         memcpy ( hessian , sptr->a , nvars * nvars * sizeof(double) ) ;
         memcpy ( grad , sptr->b , nvars * sizeof(double) ) ;
         reset_ab = 0 ;
         bad_cnt = 0 ;
         lambda *= 0.5 ;
         }

      else {
#if DEBUG
         printf ( "   BAD=%d", bad_cnt ) ;
#endif

/*
   This step caused an increase in error, so undo the step and set reset_ab
   to cause the previous Hessian and gradient to be used.  Increase lambda
   to revert closer to steepest descent (slower but more stable).
   If we had several bad iterations in a row, the Hessian may be bad, so
   increase lambda per the diagonal.  In the very unlikely event that a lot
   of bad iterations happened in a row, quit.  This should be very rare.
*/

         step_out_lm ( -1.0 , delta ) ;   // Back to original point
         reset_ab = 1 ;                   // Fetch old Hessian and gradient
         lambda *= 2.0 ;                  // Less Newton
         if (bad_cnt++ == 4) {            // If several bad in a row
            for (i=0 ; i<nvars ; i++) {   // Make sure very un-Newton
               if (hessian[i*nvars+i] > lambda)
                  lambda = hessian[i*nvars+i] ;
               }
            }
         if (bad_cnt == 10)  // Pathological escape from loop
            break ;          // Should almost never happen
         }

/*
   Diagnostic code
*/

      if (progress  &&  ++prog_cnt == 1000) {
         prog_cnt = 0 ;
         printf ( " (%lf)", error ) ;
         }
      }  // This is the end of the main iteration loop

#if DEBUG
   printf ( "\n\aLM Done=%lf  Press space...", error ) ;
   while (kbhit())
      getch() ;
   getch() ;
#endif

/*
   Free work memory
*/

CGFINISH:
   MEMTEXT ( "LEV_MARQ work" ) ;
   if (work1 != NULL)
      FREE ( work1 ) ;
   FREE ( work2 ) ;
   FREE ( hessian ) ;
   FREE ( grad ) ;
   FREE ( delta ) ;
   delete sptr ;

   return prev_err ;  // This is the best error
}

/*
--------------------------------------------------------------------------------

   Local routine to add correction vector to weight vector

--------------------------------------------------------------------------------
*/

void LayerNet::step_out_lm ( double step , double *direc )
{
   int i, n ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      for (i=0 ; i<n ; i++)
         out_coefs[i] += *(direc++) * step ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      for (i=0 ; i<n ; i++)
         hid1_coefs[i] += *(direc++) * step ;
      n = nout * nhid1_n ;
      for (i=0 ; i<n ; i++)
         out_coefs[i] += *(direc++) * step ;
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      for (i=0 ; i<n ; i++)
         hid1_coefs[i] += *(direc++) * step ;
      n = nhid2 * nhid1_n ;
      for (i=0 ; i<n ; i++)
         hid2_coefs[i] += *(direc++) * step ;
      n = nout * nhid2_n ;
      for (i=0 ; i<n ; i++)
         out_coefs[i] += *(direc++) * step ;
      }
}

/*
--------------------------------------------------------------------------------

   Local routine for debugging

--------------------------------------------------------------------------------
*/

#define DELTA 0.000001
void LayerNet::check_lm_grad ( TrainingSet *tptr , double *grad )
{
   int i, j, n ;
   double f0, f1, deriv, dot, len1, len2, factor ;

   dot = len1 = len2 = 0.0 ;
   factor = (double) tptr->ntrain * (double) nout / 50.0 ;

#if DEBUG > 2
   printf ( "\nHID1: " ) ;
#endif
   for (i=0 ; i<nhid1 ; i++) {
      for (j=0 ; j<nin_n ; j++) {
         hid1_coefs[i*nin_n+j] += DELTA ;
         f0 = trial_error ( tptr ) * factor ;
         hid1_coefs[i*nin_n+j] -= 2.0 * DELTA ;
         f1 = trial_error ( tptr ) * factor ;
         hid1_coefs[i*nin_n+j] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
#if DEBUG > 2
#if DEBUG > 3
         printf ( " (%lf: %lf %lf)", hid1_coefs[i*nin_n+j],
                    100.0 * deriv, 100.0 * *grad ) ;
#else
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * *grad ) ;
#endif
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
         f0 = trial_error ( tptr ) * factor ;
         hid2_coefs[i*nhid1_n+j] -= 2.0 * DELTA ;
         f1 = trial_error ( tptr ) * factor ;
         hid2_coefs[i*nhid1_n+j] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
#if DEBUG > 2
#if DEBUG > 3
         printf ( " (%lf: %lf %lf)", hid2_coefs[i*nhid1_n+j],
                    100.0 * deriv, 100.0 * *grad ) ;
#else
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * *grad ) ;
#endif
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
         f0 = trial_error ( tptr ) * factor ;
         out_coefs[i*n+j] -= 2.0 * DELTA ;
         f1 = trial_error ( tptr ) * factor ;
         out_coefs[i*n+j] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
#if DEBUG > 2
#if DEBUG > 3
         printf ( " (%lf: %lf %lf)", out_coefs[i*n+j],
                    100.0 * deriv, 100.0 * *grad ) ;
#else
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * *grad ) ;
#endif
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