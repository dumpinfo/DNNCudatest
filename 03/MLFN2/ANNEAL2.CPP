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
/*  ANNEAL2 - Use simulated annealing to optimize LayerNet weights            */
/*            This is the generic traditional method.                         */
/*                                                                            */
/*  This returns 1 if user pressed ESCape, -1 if insufficient memory, else 0. */
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

int LayerNet::anneal2 (
   TrainingSet *tptr ,        // Training set to use
   struct LearnParams *lptr , // User's general learning parameters
   LayerNet *worknet ,        // Work area used to keep current network
   LayerNet *bestnet ,        // Work area used to keep best network
   int init                   // Use zero suffix (initialization) anneal parms?
   )
{
   int ntemps, niters, setback, reg, nvars, user_quit ;
   int i, iter, improved, itemp, climb, reduction, n ;
   int n1, n2, n3, na1, na2, na3 ;
   char msg[80] ;
   double tempmult, temp, fval, bestfval, starttemp, stoptemp, fquit ;
   double current_fval, prob, ratio, fsum, fsqsum ;
   enum RandomDensity density ;
   SingularValueDecomp *sptr ;
   struct AnnealParams *aptr ; // User's annealing parameters
                             
   aptr = lptr->ap ;

/*
   The parameter 'init' is nonzero if we are initializing
   weights for learning.  If zero we are attempting to break
   out of a local minimum.  The main effect of this parameter
   is whether or not we use the zero suffix variables in the
   anneal parameters.
   A second effect is that regression is used only for
   initialization, not for escape (unless outputs linear.  See next).
*/

   if (init) {
      ntemps = aptr->temps0 ;
      niters = aptr->iters0 ;
      setback = aptr->setback0 ;
      starttemp = aptr->start0 ;
      stoptemp = aptr->stop0 ;
      ratio = aptr->ratio0 ;
      if (aptr->random0 == ANNEAL_GAUSSIAN)
         density = NormalDensity ;
      else if (aptr->random0 == ANNEAL_CAUCHY)
         density = CauchyDensity ;
      climb = aptr->climb0 ;
      reduction = aptr->reduc0 ;
      }
   else {
      ntemps = aptr->temps ;
      niters = aptr->iters ;
      setback = aptr->setback ;
      starttemp = aptr->start ;
      stoptemp = aptr->stop ;
      ratio = aptr->ratio ;
      if (aptr->random == ANNEAL_GAUSSIAN)
         density = NormalDensity ;
      else if (aptr->random == ANNEAL_CAUCHY)
         density = CauchyDensity ;
      climb = aptr->climb ;
      reduction = aptr->reduc ;
      }

   if (! (ntemps * niters))
      return 0 ;

/*
   Initialize other local parameters.  Note that there is no sense using
   regression if there are no hidden layers.  Also, regression is almost
   always counterproductive for local minimum escape if outputs nonlinear.
*/

   fquit = lptr->quit_err ;
   reg = (init  ||  outlin)  &&  nhid1 ;

/*
   Allocate the singular value decomposition object for REGRESS.
   Also allocate a work area for REGRESS to preserve matrix.
*/

   if (reg) {                 // False if no hidden layers
      if (nhid2 == 0)         // One hidden layer
         nvars = nhid1_n ;
      else                    // Two hidden layers
         nvars = nhid2_n ;

      i = (model == NETMOD_COMPLEX)  ?  2 * tptr->ntrain : tptr->ntrain ;

      if (i < nvars) {
         warning_message ( "Too few training sets for regression." ) ;
         reg = 0 ;
         }
      else {
         MEMTEXT ( "ANNEAL2: new SingularValueDecomp" ) ;
         sptr = new SingularValueDecomp ( i , nvars , 1 ) ;

         if ((sptr == NULL)  || ! sptr->ok) {
            memory_message (
               "for annealing with regression.  Using total randomization.");
            if (sptr != NULL)
               delete sptr ;
            reg = 0 ;
            }
         }
      }

/*
   The best point so far is kept in 'bestnet', so initialize it to the
   user's starting estimate.   Also, initialize 'bestfval', the best
   function value so far, to be the function value at that starting point.
   The current net is in 'worknet', so similarly copy this net to it.
*/

   copy_weights ( bestnet , this ) ; // Current weights are best so far
   copy_weights ( worknet , this ) ;
   bestfval = current_fval = trial_error ( tptr ) ;

/*
   If this is being used to initialize the weights, make sure that they are
   not identically zero.  Do this by setting bestfval huge so that
   SOMETHING is accepted later.
*/

   if (init  &&  nhid1) {
      i = nhid1 * nin_n ;
      while (i--) {
         if (fabs(hid1_coefs[i]) > 1.e-10)
            break ;
         }
      if (i < 0)
         bestfval = 1.e30 ;
      }

   else if (init  &&  ! nhid1) {
      i = nin_n ;
      while (i--) {
         if (fabs(out_coefs[i]) > 1.e-10)
            break ;
         }
      if (i < 0)
         bestfval = 1.e30 ;
      }

/*
   Compute the starting temperature and the factor that will be needed
   to reduce it later.
*/

   temp = starttemp ;
   if (reduction == ANNEAL_REDUCE_EXPONENTIAL)
      tempmult = exp( log( stoptemp / starttemp ) / (ntemps-1)) ;
   else if (reduction == ANNEAL_REDUCE_FAST)
      tempmult = (starttemp - stoptemp) / (stoptemp * (ntemps-1)) ;

/*
   We use a heuristic method to estimate the ratio by which the temperature
   (standard deviation of perturbations) is multiplied to get the scale
   factor for converting the function change to an acceptance probability.
   This is the user-specified ratio times the standard deviation.
   Simultaneously take this opportunity to look for a good starting point.
*/

   n = 0 ;                                // Samples going into estimate
   fsum = fsqsum = 0.0 ;                  // Sum of f and its square
   normal_message ( "Initializing..." ) ;

   for (iter=0 ; iter<(niters*ntemps/10+1) ; iter++) { // Dedicate 10%

      perturb ( worknet , this , temp ,
                reg , density ) ;         // Randomly perturb about start

      if (reg)                            // If using regression, estimate
         fval = regress ( tptr , sptr ) ; // out weights now
      else                                // Otherwise just evaluate
         fval = trial_error ( tptr ) ;

      ++n ;
      fsum += fval ;
      fsqsum += fval * fval ;

      if (fval < bestfval) {              // If this iteration improved
         copy_weights ( bestnet , this ) ; // Maintain record of best
         bestfval = fval ;                // then update the best so far

         if (bestfval <= fquit)           // If we reached the user's
            goto FINISH ;                 // limit, we can quit

         iter -= setback ;                // It often pays to keep going
         if (iter < 0)                    // at this temperature if we
            iter = 0 ;                    // are still improving
         }

      if ((user_quit = user_pressed_escape ()) != 0)
         goto FINISH ;
      }                       // Loop: for all initial iters

   fsum /= n ;                               // Mean function
   fsqsum /= n ;                             // Mean square
   fsqsum = sqrt ( fsqsum - fsum * fsum ) ;  // Standard deviation
   ratio *= fsqsum / starttemp ;             // Scaling ratio

   sprintf ( msg , " err=%.6lf  std=%.6lf", bestfval, fsqsum ) ;
   progress_message ( msg ) ;

   copy_weights ( worknet , bestnet ) ;      // The best above becomes current
   current_fval = bestfval ;
   user_quit = 0 ;                           // Flags user pressing ESCape

   for (itemp=0 ; itemp<ntemps ; itemp++) {  // Temp reduction loop

      improved = 0 ;                         // Flags if this temp improved
      n1 = n2 = n3 = na1 = na2 = na3 = 0 ;   // For acceptance rate display

      if (init) {
         sprintf ( msg , "ANNEAL2 temp=%.3lf ", temp ) ;
         normal_message ( msg ) ;
         }


      for (iter=0 ; iter<niters ; iter++) {  // Iters per temp loop

         perturb ( worknet , this , temp ,
                   reg , density ) ;         // Randomly perturb about current

         if (reg)                            // If using regression, estimate
            fval = regress ( tptr , sptr ) ; // out weights now
         else                                // Otherwise just evaluate
            fval = trial_error ( tptr ) ;

         if (fval < bestfval) {              // If this iteration improved
            copy_weights ( bestnet , this ) ; // Maintain record of best
            bestfval = fval ;                // then update the best so far
            improved = 1 ;                   // Flag that we improved

            if (bestfval <= fquit)           // If we reached the user's
               break ;                       // limit, we can quit

            iter -= setback ;                // It often pays to keep going
            if (iter < 0)                    // at this temperature if we
               iter = 0 ;                    // are still improving
            }

         prob = exp ( (current_fval - fval) / (ratio * temp) ) ;

         if (! climb)
            prob = 1.0 / (1.0 + 1.0 / prob) ;

         if (iter < 10)
            ++n1 ;
         else if (iter < niters * 3 / 4)
            ++n2 ;
         else
            ++n3 ;

         if (unifrand() < prob) {
            copy_weights ( worknet , this ) ;
            current_fval = fval ;
            if (iter < 10)
               ++na1 ;
            else if (iter < niters * 3 / 4)
               ++na2 ;
            else
               ++na3 ;
            }

         if ((user_quit = user_pressed_escape ()) != 0)
            break ;
         }                       // Loop: for all iters at a temp

      if (init) {
         sprintf ( msg , " (%.3lf %.3lf %.3lf)",
                      100.0 * na1 / (double) n1,
                      100.0 * na2 / (n2+1.e-30) ,
                      100.0 * na3 / (n3+1.e-30) ) ;
         progress_message ( msg ) ;
         }

      if (init  &&  improved) {  // If this temp saw improvement
         sprintf ( msg , " err=%.6lf", bestfval ) ;
         progress_message ( msg ) ;
         }

      if (bestfval <= fquit)  // If we reached the user's
         break ;              // limit, we can quit

      if (user_quit)
         break ;

      if (reduction == ANNEAL_REDUCE_EXPONENTIAL)
         temp *= tempmult ;
      else if (reduction == ANNEAL_REDUCE_FAST)
         temp = starttemp / (1.0 + tempmult * (itemp+1)) ;

      } // Temperature reduction loop


/*
   The trials left this weight set and neterr in random condition.
   Make them equal to the best, which will be the original
   if we never improved.
*/

FINISH:

   copy_weights ( this , bestnet ) ; // Return best weights in this net
   neterr = bestfval ;               // Trials destroyed weights, err

   if (reg) {
      MEMTEXT ( "ANNEAL2: delete SingularValueDecomp" ) ;
      delete sptr ;
      }

   if (user_quit)
      return 1 ;
   else
      return 0 ;
}
