/******************************************************************************/
/*                                                                            */
/*  LEV_MARQ - Do Levenberg-Marquardt direct descent learning                 */
/*                                                                            */
/*  Normally this returns the scaled mean square error.                       */
/*  If the user interrupted, it returns the negative mean square error.       */
/*                                                                            */
/* Copyright (c) 1995 Timothy Masters.  All rights reserved.                  */
/* Reproduction or translation of this work beyond that permitted in section  */
/* 117 of the 1976 United States Copyright Act without the express written    */
/* permission of the copyright owner is unlawful.  Requests for further       */
/* information should be addressed to the Permissions Department, John Wiley  */
/* & Sons, Inc.  The purchaser may make backup copies for his/her own use     */
/* only and not for distribution or resale.                                   */
/* Neither the author nor the publisher assumes responsibility for errors,    */
/* omissions, or damages, caused by the use of these programs or from the     */
/* use of the information contained herein.                                   */
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

double lev_marq (
   int maxits ,           // Iteration limit
   double critlim ,       // Quit if crit drops this low
   double tol ,           // Convergence tolerance
   double (*criter) (double * , double * , double * ) , // Criterion func
   int nvars ,            // Number of variables
   double *x ,            // In/out of independent variable
   SingularValueDecomp *sptr , // Work object
   double *grad ,         // Work vector n long
   double *delta ,        // Work vector n long
   double *hessian ,      // Work vector n*n long
   int progress           // Print progress?
   )
{
   int i, iter, bad_cnt, trivial_cnt, reset_ab ;
   double error, maxgrad, lambda ;
   double prev_err, improvement ;
   char msg[84] ;
   int prog_cnt=0 ;
                     
/*
   Compute the error, hessian, and error gradient at the starting point.
*/

   error = criter ( x , hessian , grad ) ;
   prev_err = error ;  // Will be 'previous iteration' error
   reset_ab = 1 ;      // Flag to use most recent good hessian and grad

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

      if (user_pressed_escape()) { // Was a key pressed?
         prev_err = -prev_err ;    // Flags user that ESCape was pressed
         break ;
         }

      if (error <= critlim)  // If our error is within user's limit
         break ;             // then we are done!

      if (error <= tol)      // Good in case converging to zero
         break ;

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

      for (i=0 ; i<nvars ; i++)
         x[i] += delta[i] ;
      error = criter ( x , sptr->a , sptr->b ) ;

#if DEBUG
      printf ( "  new=%lf", error ) ;
#if DEBUG > 3
      printf ( "\n(Dhess grad): " ) ;
      for (i=0 ; i<nvars ; i++)
         printf ( " (%lf %lf)", sptr->a[i*nvars+i], sptr->b[i] ) ;
#endif
#endif

      if (prev_err < 1.0)
         improvement = prev_err - error ;
      else 
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
         if (improvement < tol) {
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
            if (maxgrad <= tol)
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

         for (i=0 ; i<nvars ; i++)
            x[i] -= delta[i] ;
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

      if (++prog_cnt >= 1000 / nvars) {
         prog_cnt = 0 ;
         sprintf ( msg , "   LM error = %lf  lambda = %lf", prev_err, lambda ) ;
         if (progress)
            write_progress ( msg ) ;
         else 
            write_non_progress ( msg ) ;
         }
      }  // This is the end of the main iteration loop

#if DEBUG
   printf ( "\n\aLM Done=%lf  Press space...", error ) ;
   while (kbhit())
      getch() ;
   getch() ;
#endif

   return prev_err ;  // This is the best error
}
