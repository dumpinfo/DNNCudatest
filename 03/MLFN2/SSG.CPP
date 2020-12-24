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
/*  SSG - Use stochastic smoothing with gradients to learn LayerNet weights.  */
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

void LayerNet::ssg (
   TrainingSet *tptr ,        // Training set to use
   struct LearnParams *lptr , // User's general learning parameters
   int use_grad               // SS if zero, else SSG
   )
{
   int itry, user_quit, n, n_grad ;
   long seed ;
   double best_err, *work1, *work2, *grad, *avg_grad ;
   char msg[80] ;
   LayerNet *worknet1, *worknet2, *bestnet ;
                             
/*
   Allocate network scratch memory
*/

   MEMTEXT ( "SSG::new 2 worknets, bestnet" ) ;
   worknet1 = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                             nout , 0 , 0 ) ;
   worknet2 = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                             nout , 0 , 0 ) ;
   bestnet = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                            nout , 0 , 1 ) ;

   if ((worknet1 == NULL)  ||  (! worknet1->ok)
    || (worknet2 == NULL)  ||  (! worknet2->ok)
    || (bestnet == NULL)  ||  (! bestnet->ok)) {
      memory_message ( "to learn" ) ;
      if (worknet1 != NULL)
         delete worknet1 ;
      if (worknet2 != NULL)
         delete worknet2 ;
      if (bestnet != NULL)
         delete bestnet ;
      errtype = 0 ;
      return ;
      }

/*
   Allocate gradient work memory.
   Work1 is used for hidden layer 2 deltas in REAL model, and output
   activation partial derivatives and deltas in all COMPLEX models.
   Work2 is output deltas in REAL model, error difference in COMPLEX models.
*/

   if (use_grad) {
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
         MEMTEXT ( "SSG::work1" ) ;
         work1 = (double *) MALLOC ( n * sizeof(double) ) ;
         if (work1 == NULL) {
            memory_message ( "to learn" ) ;
            delete worknet1 ;
            delete worknet2 ;
            delete bestnet ;
            errtype = 0 ;
            return ;
            }
         }
      else
         work1 = NULL ;

      if (nhid1 == 0)               // No hidden layer
         n_grad = nout * nin_n ;
      else if (nhid2 == 0)          // One hidden layer
         n_grad = nhid1 * nin_n + nout * nhid1_n ;
      else                          // Two hidden layers
         n_grad = nhid1 * nin_n + nhid2 * nhid1_n + nout * nhid2_n ;

      MEMTEXT ( "SSG::3 work vectors" ) ;
      work2 = (double *) MALLOC ( nout_n * sizeof(double) ) ;
      grad = (double *) MALLOC ( n_grad * sizeof(double) ) ;
      avg_grad = (double *) MALLOC ( n_grad * sizeof(double) ) ;

      if ((work2 == NULL)  ||  (grad == NULL)  ||  (avg_grad == NULL)) {
         if (work1 != NULL)
            FREE ( work1 ) ;
         if (work2 != NULL)
            FREE ( work2 ) ;
         if (grad != NULL)
            FREE ( grad ) ;
         if (avg_grad != NULL)
            FREE ( avg_grad ) ;
         memory_message ( "to learn" ) ;
         delete worknet1 ;
         delete worknet2 ;
         delete bestnet ;
         errtype = 0 ;
         return ;
         }
      }
   else
      work1 = work2 = grad = avg_grad = NULL ;

   best_err = 1.e30 ;
   for (itry=1 ; itry<=lptr->retries+1 ; itry++) {

      user_quit = ssg_core ( tptr , lptr , worknet1 , worknet2 ,
                             work1 , work2 , grad , avg_grad , n_grad ) ;

      if (neterr < best_err) {
         best_err = neterr ;
         copy_weights ( bestnet , this ) ;
         }

      sprintf ( msg , "Try %d  err=%lf  best=%lf", itry, neterr, best_err ) ;
      normal_message ( msg ) ;

      if (user_quit  ||  (neterr < lptr->quit_err))
         break ;

      seed = flrand() - (long) (itry * 97) ;   // Insure new seed for anneal
      sflrand ( seed ) ;
      zero_weights () ;  // Retry random
      }

   copy_weights ( this , bestnet ) ;
   neterr = best_err ;

   MEMTEXT ( "AN1::learn delete 2 worknets, bestnet" ) ;
   delete worknet1 ;
   delete worknet2 ;
   delete bestnet ;

   if (use_grad) {
      if (work1 != NULL) {
         MEMTEXT ( "SSG::work1" ) ;
         FREE ( work1 ) ;
         }
      MEMTEXT ( "SSG::3 work vectors" ) ;
      FREE ( work2 ) ;
      FREE ( grad ) ;
      FREE ( avg_grad) ;
      }

   return ;

}

int LayerNet::ssg_core (
   TrainingSet *tptr ,        // Training set to use
   struct LearnParams *lptr , // User's general learning parameters
   LayerNet *avgnet ,         // Work area used to keep average weights
   LayerNet *bestnet ,        // And the best so far
   double *work1 ,            // Gradient work vector
   double *work2 ,            // Ditto
   double *grad ,             // Ditto
   double *avg_grad ,         // Ditto
   int n_grad                 // Length of above vectors
   )
{
   int ntemps, niters, setback, reg, nvars, user_quit ;
   int i, iter, itemp, n_good, n_bad, use_grad ;
   char msg[80] ;
   double tempmult, temp, fval, bestfval, starttemp, stoptemp, fquit ;
   double avg_func, new_fac, gradlen, grad_weight, weight_used ;
   enum RandomDensity density ;
   SingularValueDecomp *sptr ;
   struct AnnealParams *aptr ; // User's annealing parameters

   aptr = lptr->ap ;

   ntemps = aptr->temps0 ;
   niters = aptr->iters0 ;
   setback = aptr->setback0 ;
   starttemp = aptr->start0 ;
   stoptemp = aptr->stop0 ;
   if (aptr->random0 == ANNEAL_GAUSSIAN)
      density = NormalDensity ;
   else if (aptr->random0 == ANNEAL_CAUCHY)
      density = CauchyDensity ;

   if (! (ntemps * niters))
      return 0 ;

/*
   Initialize other local parameters.  Note that there is no sense using
   regression if there are no hidden layers.
*/

   use_grad = (grad != NULL) ;
   fquit = lptr->quit_err ;
   reg = nhid1 ;

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
         MEMTEXT ( "SSG: new SingularValueDecomp" ) ;
         sptr = new SingularValueDecomp ( i , nvars , 1 ) ;

         if ((sptr == NULL)  || ! sptr->ok) {
            memory_message (
               "for SS(G) with regression.  Using total randomization.");
            if (sptr != NULL)
               delete sptr ;
            reg = 0 ;
            }
         }
      }

/*
   For the basic algorithm, we will keep the current 'average' network
   weight set in avgnet.  This will be the moving center about which the
   perturbation is done.
   Although not directly related to the algorithm itself, we keep track
   of the best network ever found in bestnet.  That is what the user
   will get at the end.
*/

   copy_weights ( bestnet , this ) ; // Current weights are best so far
   copy_weights ( avgnet , this ) ;  // Center of perturbation
   bestfval = trial_error ( tptr ) ;

/*
   If this is being used to initialize the weights, make sure that they are
   not identically zero.  Do this by setting bestfval huge so that
   SOMETHING is accepted later.
*/

   if (nhid1) {
      i = nhid1 * nin_n ;
      while (i--) {
         if (fabs(hid1_coefs[i]) > 1.e-10)
            break ;
         }
      if (i < 0)
         bestfval = 1.e30 ;
      }

/*
   Initialize by cumulating a bunch of points
*/

   normal_message ( "Initializing..." ) ;
   avg_func = 0.0 ;                       // Mean function around center
   if (use_grad) {
      for (i=0 ; i<n_grad ; i++)          // Zero the mean gradient
         avg_grad[i] = 0.0 ;
      }

   for (iter=0 ; iter<niters ; iter++) {  // Initializing iterations

      perturb ( avgnet , this , starttemp , reg , density ) ; // Move point

      if (reg)                            // If using regression, estimate
         fval = regress ( tptr , sptr ) ; // out weights now, ignore fval
      if (use_grad)                       // Also need gradient?
         fval = gradient ( tptr , work1 , work2 , grad ) ; // fval redundant
      else if (! reg)                     // If reg we got fval from regress
         fval = trial_error ( tptr ) ;

      avg_func += fval ;                  // Cumulate mean function

      if (use_grad) {                     // Also need gradient?
         for (i=0 ; i<n_grad ; i++)       // Cumulate mean gradient
            avg_grad[i] += grad[i] ;
         }

      if (fval < bestfval) {              // If this iteration improved
         bestfval = fval ;                // then update the best so far
         copy_weights ( bestnet , this ) ; // Keep the network
         if (bestfval <= fquit)           // If we reached the user's
            goto FINISH ;                 // limit, we can quit
         }

      if ((user_quit = user_pressed_escape ()) != 0)
         goto FINISH ;

      } // Loop: for all initial iters

   avg_func /= niters ;          // Mean of all points around avgnet
   new_fac = 1.0 / niters ;      // Weight of each point

   sprintf ( msg , "  avg=%.6lf  best=%.6lf", avg_func, bestfval ) ;
   progress_message ( msg ) ;

   if (use_grad) {               // Also need gradient?
      gradlen = 0.0 ;            // Will cumulate grad length
      for (i=0 ; i<n_grad ; i++) {  // Find gradient mean and length
         avg_grad[i] /= niters ;
         gradlen += avg_grad[i] * avg_grad[i] ;
         }
      gradlen = sqrt ( gradlen ) ;
      grad_weight = 0.5 ;
      }

/*
   This is the temperature reduction loop and the iteration within
   temperature loop.
*/

   temp = starttemp ;
   tempmult = exp( log( stoptemp / starttemp ) / (ntemps-1)) ;
   user_quit = 0 ;                           // Flags user pressed ESCape

   for (itemp=0 ; itemp<ntemps ; itemp++) {  // Temp reduction loop

      n_good = n_bad = 0 ;                   // Counts better and worse

      sprintf ( msg , "Temp=%.3lf ", temp ) ;
      normal_message ( msg ) ;

      for (iter=0 ; iter<niters ; iter++) {  // Iters per temp loop

         if ((n_bad >= 10)  &&
             ((double) n_good / (double) (n_good+n_bad)  <  0.15))
            break ;

         perturb ( avgnet , this , temp ,
                   reg , density ) ;         // Randomly perturb about center

         if (use_grad)                       // Bias per gradient?
            weight_used = shift ( grad , this , grad_weight , reg ) ;

         if (reg) {                          // If using regression, estimate
            fval = regress ( tptr , sptr ) ; // out weights now
            if ((user_quit = user_pressed_escape ()) != 0)
               break ;
            if (fval >= avg_func) {          // If this would raise mean
               ++n_bad ;                     // Count this bad point for user
               continue ;                    // Skip it and try again
               }
            }

         if (use_grad)                       // Need gradient, fval redundant
            fval = gradient ( tptr , work1 , work2 , grad ) ;
         else if (! reg)                     // If reg we got fval from regress
            fval = trial_error ( tptr ) ;

         if ((user_quit = user_pressed_escape ()) != 0)
            break ;

         if (fval >= avg_func) {             // If this would raise mean
            ++n_bad ;                        // Count this bad point for user
            continue ;                       // Skip it and try again
            }

         ++n_good ;

         if (fval < bestfval) {              // If this iteration improved
            bestfval = fval ;                // then update the best so far
            copy_weights ( bestnet , this ) ; // Keep the network

            if (bestfval <= fquit)           // If we reached the user's
               break ;                       // limit, we can quit

            iter -= setback ;                // It often pays to keep going
            if (iter < 0)                    // at this temperature if we
               iter = 0 ;                    // are still improving
            }

         adjust ( avgnet , this , reg , new_fac ) ; // Move center slightly
         avg_func = new_fac * fval  +  (1.0 - new_fac) * avg_func ;
         if (use_grad) {
            grad_weight = new_fac * weight_used + (1.0 - new_fac) * grad_weight ;
            for (i=0 ; i<n_grad ; i++)          // Adjust mean gradient
               avg_grad[i] = new_fac * grad[i] + (1.0 - new_fac) * avg_grad[i] ;
            }
         }                                   // Loop: for all iters at a temp

/*
   Iters within temp loop now complete
*/

      sprintf ( msg , " %.3lf%% improved  avg=%.5lf  best=%.5lf",
         100.0 * n_good / (double) (n_good+n_bad), avg_func, bestfval ) ;
      progress_message ( msg ) ;

      if (use_grad) {
         gradlen = 0.0 ;                        // Will cumulate grad length
         for (i=0 ; i<n_grad ; i++)             // Find gradient length
            gradlen += avg_grad[i] * avg_grad[i] ;
         gradlen = sqrt ( gradlen ) ;
         sprintf ( msg , "  grad=%.5lf", gradlen ) ;
         progress_message ( msg ) ;
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
*/

FINISH:
   copy_weights ( this , bestnet ) ; // Return best weights in this net
   neterr = bestfval ;               // Trials destroyed weights, err

   if (reg) {
      MEMTEXT ( "SSG: delete SingularValueDecomp" ) ;
      delete sptr ;
      }

   if (user_quit)
      return 1 ;
   else
      return 0 ;
}

/*
--------------------------------------------------------------------------------

   shift - Shift the weights toward the (negative) gradient

--------------------------------------------------------------------------------
*/

double LayerNet::shift( double *grad , LayerNet *pert , double weight , int reg)
{
   int n ;
   double length, *outgrad, *hid1grad, *hid2grad, x1, x2 ;

/*
   Compute individual gradient positions in total vector.
*/

   if (nhid1 == 0)        // No hidden layer
      outgrad = grad ;
   else if (nhid2 == 0) { // One hidden layer
      hid1grad = grad ;
      outgrad = grad + nhid1 * nin_n ;
      }
   else {                 // Two hidden layers
      hid1grad = grad ;
      hid2grad = grad + nhid1 * nin_n ;
      outgrad = hid2grad + nhid2 * nhid1_n ;
      }

/*
   Compute the effective length of the gradient vector.
   This is the weight parameter times a positive random variable with unit mean.
*/

   normal_pair ( &x1 , &x2 ) ;
   length = x1 * x1 + x2 * x2 ;
   normal_pair ( &x1 , &x2 ) ;
   length += x1 * x1 + x2 * x2 ;
   length *= 0.25 * weight ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      while (n--)
         pert->out_coefs[n] += outgrad[n] * length ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      while (n--)
         pert->hid1_coefs[n] += hid1grad[n] * length ;
      if (! reg) {
         n = nout * nhid1_n ;
         while (n--)
            pert->out_coefs[n] += outgrad[n] * length ;
         }
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      while (n--)
         pert->hid1_coefs[n] += hid1grad[n] * length ;
      n = nhid2 * nhid1_n ;
      while (n--)
         pert->hid2_coefs[n] += hid2grad[n] * length ;
      if (! reg) {
         n = nout * nhid2_n ;
         while (n--)
            pert->out_coefs[n] += outgrad[n] * length ;
         }
      }

   return length ;
}

/*
--------------------------------------------------------------------------------

   adjust - slightly move the center

--------------------------------------------------------------------------------
*/

void LayerNet::adjust ( LayerNet *cent , LayerNet *pert , int reg , double fac )
{
   int n ;
   double old ;

   old = 1.0 - fac ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      while (n--)
         cent->out_coefs[n] = old*cent->out_coefs[n] + fac*pert->out_coefs[n] ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      while (n--)
         cent->hid1_coefs[n]= old*cent->hid1_coefs[n] + fac*pert->hid1_coefs[n];
      if (! reg) {
         n = nout * nhid1_n ;
         while (n--)
            cent->out_coefs[n] = old*cent->out_coefs[n] + fac*pert->out_coefs[n] ;
         }
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      while (n--)
         cent->hid1_coefs[n] = old*cent->hid1_coefs[n] + fac*pert->hid1_coefs[n] ;
      n = nhid2 * nhid1_n ;
      while (n--)
         cent->hid2_coefs[n] = old*cent->hid2_coefs[n] + fac*pert->hid2_coefs[n] ;
      if (! reg) {
         n = nout * nhid2_n ;
         while (n--)
            cent->out_coefs[n] = old*cent->out_coefs[n] + fac*pert->out_coefs[n] ;
         }
      }
}
