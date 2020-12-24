/******************************************************************************/
/*                                                                            */
/*  ANNEAL2 - Use simulated annealing to optimize MLFN weights                */
/*            This is the generic traditional method.                         */
/*                                                                            */
/*  This returns the best function value, or its negative if the user         */
/*  pressed ESCape.                                                           */
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

#define DEBUG 0

double anneal2 (
   int n ,                        // This many parameters to optimize
   double *x ,                    // They are input/output here
   double *current ,              // Work vector n long
   double *best ,                 // Work vector n long
   double (*criter) ( double * ) , // Criterion func
   double min_func ,              // Starting function value, huge to force
   int temperatures ,             // Number of temperatures
   int iterations ,               // Iterations at each temperature
   int iter_reset ,               // Set back iteration counter with improvement
   double temp_start ,            // Starting temperature
   double temp_end ,              // Ending temperature
   enum RandomDensity density ,   // Density for perturbation
   double ratio ,                 // For starting probability
   int climb ,                    // Hill climbing?
   int reduction ,                // Temp reduction (EXPONENTIAL or FAST)
   double ok_error ,              // Quit if criterion drops this low
   int progress                   // Write progress?
   )
{
   int user_quit, iter, itemp ;
   int n0 ;
//   int n1, n2, n3, na1, na2, na3 ;
   char msg[400] ;
   double tempmult, temp, this_f ;
   double current_this_f, prob, fsum, fsqsum ;

/*
   The best point so far is kept in 'best', so initialize it to the
   user's starting estimate.
   The current net is in 'current', so similarly copy this net to it.
*/

   memcpy ( best , x , n * sizeof(double) ) ;
   memcpy ( current , x , n * sizeof(double) ) ;

   temp = temp_start ;
   if (temperatures < 2)
      tempmult = 1.0 ;
   else if (reduction == ANNEAL_REDUCE_EXPONENTIAL)
      tempmult = exp( log( temp_end / temp_start ) / (temperatures-1)) ;
   else if (reduction == ANNEAL_REDUCE_FAST)
      tempmult = (temp_start - temp_end) / (temp_end * (temperatures-1)) ;

/*
   We use a heuristic method to estimate the ratio by which the temperature
   (standard deviation of perturbations) is multiplied to get the scale
   factor for converting the function change to an acceptance probability.
   This is the user-specified ratio times the standard deviation.
   Simultaneously take this opportunity to look for a good starting point.
*/

   n0 = 0 ;                               // Samples going into estimate
   fsum = fsqsum = 0.0 ;                  // Sum of f and its square

   if (progress)
      write_progress ( "Initializing..." ) ;
   else 
      write_non_progress ( "Initializing..." ) ;

   for (iter=0 ; iter<(iterations*temperatures/10+10) ; iter++) { // Dedicate 10%

      shake ( n , current , x , temp , density ) ; // Randomly perturb
      this_f = criter ( x ) ;          // Compute criterion function
      ++n0 ;                           // Count evaluations
      fsum += this_f ;                 // Sum for mean
      fsqsum += this_f * this_f ;      // And standard deviation

#if DEBUG
      printf ( " g=%.4lf", this_f ) ;
#endif

      if (this_f < min_func) {            // If this iteration improved
#if DEBUG
         printf ( " BEST=%.4lf", this_f ) ;
#endif
         memcpy ( best , x , n * sizeof(double) ) ;
         min_func = this_f ;              // then update the best so far

         if (min_func <= ok_error)        // If we reached the user's
            goto FINISH ;                 // limit, we can quit

         iter -= iter_reset ;             // It often pays to keep going
         if (iter < 0)                    // at this temperature if we
            iter = 0 ;                    // are still improving
         }

      if ((user_quit = user_pressed_escape ()) != 0)
         goto FINISH ;
      }                       // Loop: for all initial iters

   fsum /= n0 ;                              // Mean function
   fsqsum /= n0 ;                            // Mean square
   fsqsum = sqrt ( fsqsum - fsum * fsum ) ;  // Standard deviation
   ratio *= fsqsum / temp_start ;            // Scaling ratio

   sprintf ( msg , " init err=%.6lf  std=%.6lf", min_func, fsqsum ) ;
   if (progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;

   memcpy ( current , best , n * sizeof(double) ) ;// The best becomes current
   current_this_f = min_func ;
   user_quit = 0 ;                           // Flags user pressing ESCape

   for (itemp=0 ; itemp<temperatures ; itemp++) {  // Temp reduction loop

#if 0
      n1 = n2 = n3 = na1 = na2 = na3 = 0 ;   // For acceptance rate display
#endif

      for (iter=0 ; iter<iterations ; iter++) {  // Iters per temp loop

         shake ( n , current , x , temp , density ) ; // Randomly perturb
         this_f = criter ( x ) ;            // Compute criterion function

#if DEBUG
         printf ( " f=%.4lf", this_f ) ;
#endif

         if (this_f < min_func) {            // If this iteration improved
#if DEBUG
            printf ( " BEST=%.4lf", this_f ) ;
#endif
            memcpy ( best , x , n * sizeof(double) ) ;
            min_func = this_f ;              // then update the best so far

            if (min_func <= ok_error)        // If we reached the user's
               break ;                       // limit, we can quit

            iter -= iter_reset ;             // It often pays to keep going
            if (iter < 0)                    // at this temperature if we
               iter = 0 ;                    // are still improving
            }

         prob = exp ( (current_this_f - this_f) / (ratio * temp) ) ;

         if (! climb)
            prob = 1.0 / (1.0 + 1.0 / prob) ;

#if DEBUG
         printf ( " p=%.3lf", prob ) ;
#endif
#if 0
         if (iter < 10)
            ++n1 ;
         else if (iter < iterations * 3 / 4)
            ++n2 ;
         else
            ++n3 ;
#endif

         if (unifrand() < prob) {
#if DEBUG
            printf ( " moved" ) ;
#endif
            memcpy ( current , x , n * sizeof(double) ) ;
            current_this_f = this_f ;
#if 0
            if (iter < 10)
               ++na1 ;
            else if (iter < iterations * 3 / 4)
               ++na2 ;
            else
               ++na3 ;
#endif
            }

         if ((user_quit = user_pressed_escape ()) != 0)
            break ;
         }                       // Loop: for all iters at a temp

      sprintf ( msg , "ANNEAL temp=%.2lf  err=%.6lf ", temp, min_func ) ;
      if (progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      if (min_func <= ok_error)  // If we reached the user's
         break ;              // limit, we can quit

      if (user_quit)
         break ;

      if (reduction == ANNEAL_REDUCE_EXPONENTIAL)
         temp *= tempmult ;
      else if (reduction == ANNEAL_REDUCE_FAST)
         temp = temp_start / (1.0 + tempmult * (itemp+1)) ;

      } // Temperature reduction loop


/*
   The trials left this weight set and neterr in random condition.
   Make them equal to the best, which will be the original
   if we never improved.
*/

FINISH:

   memcpy ( x , best , n * sizeof(double) ) ;

   if (user_quit)
      return -min_func ;
   else
      return min_func ;
}
