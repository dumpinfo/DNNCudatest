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
/*  ANNEAL1 - Use simulated annealing to optimize LayerNet weights            */
/*            This is the primitive random search for local minima.           */
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

int LayerNet::anneal1 (
   TrainingSet *tptr ,        // Training set to use
   struct LearnParams *lptr , // User's general learning parameters
   LayerNet *bestnet ,        // Work area used to keep best network
   int init ,                 // Use zero suffix (initialization) anneal parms?
   int itry                   // Passed by caller for more randomization
   )
{
   int ntemps, niters, setback, reg, nvars, user_quit ;
   int i, iter, improved, ever_improved, itemp ;
   long seed, bestseed ;
   char msg[80] ;
   double tempmult, temp, fval, bestfval, starttemp, stoptemp, fquit ;
   enum RandomDensity density ;
   SingularValueDecomp *sptr ;
   struct AnnealParams *aptr ; // User's annealing parameters
                             
   aptr = lptr->ap ;

/*
   The parameter 'init' is nonzero if we are initializing
   weights for learning.  If zero we are attempting to break
   out of a local minimum.  The main effect  of this parameter
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
      if (aptr->random0 == ANNEAL_GAUSSIAN)
         density = NormalDensity ;
      else if (aptr->random0 == ANNEAL_CAUCHY)
         density = CauchyDensity ;
      }
   else {
      ntemps = aptr->temps ;
      niters = aptr->iters ;
      setback = aptr->setback ;
      starttemp = aptr->start ;
      stoptemp = aptr->stop ;
      if (aptr->random == ANNEAL_GAUSSIAN)
         density = NormalDensity ;
      else if (aptr->random == ANNEAL_CAUCHY)
         density = CauchyDensity ;
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
         MEMTEXT ( "ANNEAL1: new SingularValueDecomp" ) ;
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
   For every temperature, the center around which we will perturb is the
   best point so far.  This is kept in 'bestnet', so initialize it to the
   user's starting estimate.   Also, initialize 'bestfval', the best
   function value so far, to be the function value at that starting point.
*/

   copy_weights ( bestnet , this ) ; // Current weights are best so far
   bestfval = trial_error ( tptr ) ;

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
   This is the temperature reduction loop and the iteration within
   temperature loop.  We use a slick trick to keep track of the
   best point at a given temperature.  We certainly don't want to
   replace the best every time an improvement is had, as then we
   would be moving our center about, compromising the global nature
   of the algorithm.  We could, of course, have a second work area
   in which we save the 'best so far for this temperature' point.
   But if there are a lot of variables, the usual case, this wastes
   memory.  What we do is to save the seed of the random number
   generator which created the improvement.  Then later, when we
   need to retrieve the best, simply set the random seed and
   regenerate it.  This technique also saves a lot of copying time
   if many improvements are made for a single temperature.
*/

   temp = starttemp ;
   tempmult = exp( log( stoptemp / starttemp ) / (ntemps-1)) ;
   ever_improved = 0 ;                       // Flags if improved at all
   user_quit = 0 ;                           // Flags user pressed ESCape

   for (itemp=0 ; itemp<ntemps ; itemp++) {  // Temp reduction loop

      improved = 0 ;                         // Flags if this temp improved

      if (init) {
         sprintf ( msg , "\nANNEAL temp=%.2lf ", temp ) ;
         progress_message ( msg ) ;
         }

      for (iter=0 ; iter<niters ; iter++) {  // Iters per temp loop

         seed = flrand () ;                  // Get a random seed
         sflrand ( seed ) ;                  // Brute force set it
         for (i=itry ; i ; i--)              // Peel off a few
            (void) flrand() ;                // To avoid reps across tries
         perturb ( bestnet , this , temp ,
                   reg , density ) ;         // Randomly perturb about best

         if (reg)                            // If using regression, estimate
            fval = regress ( tptr , sptr ) ; // out weights now
         else                                // Otherwise just evaluate
            fval = trial_error ( tptr ) ;

         if (fval < bestfval) {              // If this iteration improved
            bestfval = fval ;                // then update the best so far
            bestseed = seed ;                // and save seed to recreate it
            ever_improved = improved = 1 ;   // Flag that we improved

            if (bestfval <= fquit)           // If we reached the user's
               break ;                       // limit, we can quit

            iter -= setback ;                // It often pays to keep going
            if (iter < 0)                    // at this temperature if we
               iter = 0 ;                    // are still improving
            }

         if ((user_quit = user_pressed_escape ()) != 0)
            break ;
         }                                   // Loop: for all iters at a temp

      if (improved) {                        // If this temp saw improvement
         sflrand ( bestseed ) ;              // set seed to what caused it
         for (i=itry ; i ; i--)              // Peel off a few
            (void) flrand() ;                // To avoid reps across tries
         perturb ( bestnet , this , temp ,
                   reg , density ) ;         // Recreate best
         copy_weights ( bestnet , this ) ;   // which will become next center
         sflrand ( bestseed / 2L + 999L ) ;  // Jog seed away from best

         if (init) {
            sprintf ( msg , " err=%.6lf ", bestfval ) ;
            progress_message ( msg ) ;
            }
         }

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

   Also, if we improved and are using regression, recall that bestnet
   only contains the best hidden weights, as we did not bother to run
   regress when we updated bestnet.  Do that now before returning.
*/

   copy_weights ( this , bestnet ) ; // Return best weights in this net
   neterr = bestfval ;               // Trials destroyed weights, err

   if (ever_improved  &&  reg)
      neterr = regress ( tptr , sptr ) ; // regressed output weights

   if (reg) {
      MEMTEXT ( "ANNEAL1: delete SingularValueDecomp" ) ;
      delete sptr ;
      }

   if (user_quit)
      return 1 ;
   else
      return 0 ;
}
