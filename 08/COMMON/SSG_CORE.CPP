/******************************************************************************/
/*                                                                            */
/*  SSG_CORE - Stochastic smoothing with optional gradients                   */
/*                                                                            */
/*  This returns 1 if user pressed ESCape, -1 if insufficient memory, else 0. */
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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

double ssg_core (
   int n ,                        // This many parameters to optimize
   double *x ,                    // They are input/output here
   double (*criter)( double * , double , int , double * ) , // Criterion func
   double bestfval ,              // Starting function value, huge to force
   int ntemps ,                   // Number of temperatures
   int niters ,                   // Iterations at each temperature
   int setback ,                  // Set back iteration counter with improvement
   double starttemp ,             // Starting temperature
   double stoptemp ,              // Stopping temperature
   enum RandomDensity density ,   // Density for perturbation
   double fquit ,                 // Quit if criterion drops this low
   int use_grad ,                 // Use gradient hinting?
   double *avg ,                  // Center kept here
   double *best ,                 // Best parameters here
   double *grad ,                 // Gradient work vector
   double *avg_grad ,             // Ditto
   int progress                   // Write progress?
   )
{
   int user_quit ;
   int i, iter, itemp, n_good, n_bad ;
   char msg[80] ;
   double tempmult, temp, fval ;
   double avg_func, new_fac, gradlen, grad_weight, weight_used ;

/*
   For the basic algorithm, we will keep the current 'average' parameters
   weight set in 'avg'.  This will be the moving center about which the
   perturbation is done.
   Although not directly related to the algorithm itself, we keep track
   of the best parameters ever found in 'best'.  That is what the user
   will get at the end.
*/

   memcpy ( best , x , n * sizeof(double) ) ; // Current weights are best so far
   memcpy ( avg , x , n * sizeof(double) ) ;  // Center of perturbation

/*
   Initialize by cumulating a bunch of points
*/

   if (progress)
      write_progress ( "Initializing..." ) ;
   else 
      write_non_progress ( "Initializing..." ) ;

   avg_func = 0.0 ;                       // Mean function around center
   if (use_grad) {
      for (i=0 ; i<n ; i++)               // Zero the mean gradient
         avg_grad[i] = 0.0 ;
      }

   for (iter=0 ; iter<niters ; iter++) {  // Initializing iterations

      shake ( n , avg , x , starttemp , density ) ;  // Randomly perturb
      fval = criter ( x , 1.e90 , use_grad , grad) ; // Compute criterion, grad

      user_quit = (fval < 0.0) ;          // User pressed ESCape?
      fval = fabs ( fval ) ;

      avg_func += fval ;                  // Cumulate mean function

      if (use_grad) {                     // Also need gradient?
         for (i=0 ; i<n ; i++)            // Cumulate mean gradient
            avg_grad[i] += grad[i] ;
         }

      if (fval < bestfval) {              // If this iteration improved
         bestfval = fval ;                // then update the best so far
         memcpy ( best , x , n * sizeof(double) ) ; // Keep the network
         if (bestfval <= fquit)           // If we reached the user's
            goto FINISH ;                 // limit, we can quit
         }

      if (user_quit || user_pressed_escape ())
         goto FINISH ;

      } // Loop: for all initial iters

   avg_func /= niters ;             // Mean of all points around avgnet
   new_fac = 1.0 / niters ;         // Weight of each point

   if (use_grad) {                  // Also need gradient?
      gradlen = 0.0 ;               // Will cumulate grad length
      for (i=0 ; i<n ; i++) {       // Find gradient mean and length
         avg_grad[i] /= niters ;
         gradlen += avg_grad[i] * avg_grad[i] ;
         }
      gradlen = sqrt ( gradlen ) ;
      grad_weight = 0.5 ; // Arbitrary heuristic good for neural networks
      sprintf ( msg , "  avg=%.6lf  best=%.6lf  grad=%.6lf",
         avg_func, bestfval, gradlen ) ;
      }
   else
      sprintf ( msg , "  avg=%.6lf  best=%.6lf", avg_func, bestfval ) ;

   if (progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;

/*
   This is the temperature reduction loop and the iteration within
   temperature loop.
*/

   temp = starttemp ;
   tempmult = exp( log( stoptemp / starttemp ) / (ntemps-1)) ;
   user_quit = 0 ;                           // Flags user pressed ESCape

   for (itemp=0 ; itemp<ntemps ; itemp++) {  // Temp reduction loop

      n_good = n_bad = 0 ;                   // Counts better and worse

      for (iter=0 ; iter<niters ; iter++) {  // Iters per temp loop

         if ((n_bad >= 12)  &&
             ((double) n_good / (double) (n_good+n_bad)  <  0.2))
            break ;

         shake ( n , avg , x , starttemp , density ) ; // Randomly perturb

         if (use_grad)                       // Bias per gradient?
            weight_used = shift ( grad , x , grad_weight , n ) ;

         fval = criter ( x , avg_func , use_grad , grad ) ; // Criterion, grad

         user_quit = (fval < 0.0) ;          // User pressed ESCape?
         fval = fabs ( fval ) ;

         if (user_quit)
            break ;

         if ((user_quit = user_pressed_escape ()) != 0)
            break ;

         if (fval >= avg_func) {             // If this would raise mean
            ++n_bad ;                        // Count this bad point for user
            continue ;                       // Skip it and try again
            }

         ++n_good ;

         if (fval < bestfval) {              // If this iteration improved
            bestfval = fval ;                // then update the best so far
            memcpy ( best , x , n * sizeof(double) ) ; // Keep the network

            if (bestfval <= fquit)           // If we reached the user's
               break ;                       // limit, we can quit

            iter -= setback ;                // It often pays to keep going
            if (iter < 0)                    // at this temperature if we
               iter = 0 ;                    // are still improving
            }

/*
   Adjust everything slightly per this improved point
*/

         avg_func = new_fac * fval  +  (1.0 - new_fac) * avg_func ;
         for (i=0 ; i<n ; i ++)          // Adjust mean gradient
            avg[i] = new_fac * x[i] + (1.0 - new_fac) * avg[i] ;
         if (use_grad) {
            grad_weight = new_fac * weight_used + (1.0 - new_fac) * grad_weight;
            for (i=0 ; i<n ; i ++)          // Adjust mean gradient
               avg_grad[i] = new_fac * grad[i] + (1.0 - new_fac) * avg_grad[i] ;
            }
         }                                   // Loop: for all iters at a temp

/*
   Iters within temp loop now complete
*/

      if (use_grad) {
         gradlen = 0.0 ;                     // Will cumulate grad length
         for (i=0 ; i<n ; i++)               // Find gradient length
            gradlen += avg_grad[i] * avg_grad[i] ;
         gradlen = sqrt ( gradlen ) ;
         sprintf (msg,
             "Temp=%.3lf  %.3lf%% improved  avg=%.6lf  best=%.6lf  grad=%.6lf",
             temp, 100.0 * n_good / (n_good+n_bad+1.e-30),
             avg_func, bestfval, gradlen ) ;
         }
      else 
         sprintf ( msg , "Temp=%.3lf  %.3lf%% improved  avg=%.6lf  best=%.6lf",
            temp, 100.0 * n_good / (n_good+n_bad+1.e-30),
            avg_func, bestfval ) ;

      if (progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      if (bestfval <= fquit)  // If we reached the user's
         break ;              // limit, we can quit

      if (user_quit)
         break ;

      temp *= tempmult ;      // Reduce temp for next pass
      }                       // through this temperature loop


/*
   The trials left this weight set and neterr in random condition.
   Make them equal to the best, which will be the original
   if we never improved.
*/

FINISH:
   memcpy ( x , best , n * sizeof(double) ) ;

   if (user_quit)
      return -bestfval ;
   else
      return bestfval ;
}

/*
--------------------------------------------------------------------------------

   shift - Shift the weights toward the (negative) gradient

--------------------------------------------------------------------------------
*/

double shift( double *grad , double *pert , double weight , int n )
{
   double length, x1, x2 ;

/*
   Compute the effective length of the gradient vector.
   This is the weight parameter times a positive random variable with unit mean.
   We use a chi-square with 4 degrees of freedom
*/

   normal_pair ( &x1 , &x2 ) ;
   length = x1 * x1 + x2 * x2 ;
   normal_pair ( &x1 , &x2 ) ;
   length += x1 * x1 + x2 * x2 ;
   length *= 0.25 * weight ;

   while (n--)
      pert[n] += grad[n] * length ;

   return length ;
}
