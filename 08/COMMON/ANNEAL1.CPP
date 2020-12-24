/******************************************************************************/
/*                                                                            */
/*  ANNEAL1 - Use simulated annealing to optimize MLFN weights                */
/*            This is the primitive random search for local minima.           */
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

double anneal1 (
   int n ,                        // This many parameters to optimize
   double *x ,                    // They are input/output here
   double *work ,                 // Work vector n long
   double (*criter) ( double * ) , // Criterion func
   double min_func ,              // Starting function value, huge to force
   int itry ,                     // Passed by caller for more randomization
   int temperatures ,             // Number of temperatures
   int iterations ,               // Iterations at each temperature
   int iter_reset ,               // Set back iteration counter with improvement
   double temp_start ,            // Starting temperature
   double temp_end ,              // Stopping temperature
   enum RandomDensity density ,   // Density for perturbation
   double ok_error ,              // Quit if criterion drops this low
   int progress                   // Write progress?
   )
{
   int user_quit ;
   int i, iter, improved, itemp ;
   long seed, saveseed ;
   char msg[400] ;
   double tempmult, temp, this_f ;
                             
/*
   We shake around a center of 'x'.
   Instead of saving the best point, just save the random seed that
   created it!
*/

   temp = temp_start ;
   if (temperatures < 2)
      tempmult = 1.0 ;
   else
      tempmult = exp( log( temp_end / temp_start ) / (temperatures-1)) ;
   user_quit = 0 ;                           // Flags user pressed ESCape

   for (itemp=0 ; itemp<temperatures ; itemp++) {  // Temperature reduction loop
      improved = 0 ;                               // Flags improvement

      for (iter=0 ; iter<iterations ; iter++) {    // Iters within temperature
         seed = flrand () ;                  // Get a new random number
         sflrand ( seed ) ;                  // Set seed to insure repeats
         for (i=itry ; i ; i--)              // Peel off a few
            (void) flrand() ;                // To avoid reps across tries
         shake ( n , x , work , temp , density ) ; // Randomly perturb

         this_f = criter ( work ) ;          // Compute criterion function

         if (this_f < min_func) {            // If this trial improved
            improved = 1 ;                   // Flag improvement
            iter -= iter_reset ;             // Set back iteration counter
            if (iter < 0)                    // To keep trying here
               iter = 0 ;                    // Insurance

            min_func = this_f ;              // Update record best
            saveseed = seed ;                // Save seed to reproduce

            if (min_func <= ok_error)        // If at user's limit
               break ;                       // We can quit
            }

         if ((user_quit = user_pressed_escape ()) != 0)
            break ;
         }                                   // Loop: for all iters at a temp

      if (improved) {                        // If improvement here
         sflrand ( saveseed ) ;              // set seed to reproduce point
         for (i=itry ; i ; i--)              // Peel off a few
            (void) flrand() ;                // To avoid reps across tries
         shake ( n , x , work , temp , density ) ; // Recreate best
         memcpy ( x , work , n * sizeof(double)) ; // which is next center
         sflrand ( saveseed / 4 + 17 ) ;     // Change seed
         }

      sprintf ( msg , "ANNEAL temp=%.2lf  err=%.6lf ", temp, min_func ) ;
      if (progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      if (min_func <= ok_error)  // If at user's limit
         break ;                 // We can quit

      if (user_quit)
         break ;

      temp *= tempmult ;      // Reduce the temperature for the next pass
      }                       // through this temperature loop

   if (user_quit)
      return -min_func ;
   else
      return min_func ;
}
