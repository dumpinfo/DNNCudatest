/******************************************************************************/
/*                                                                            */
/*  SSG - Use stochastic smoothing with gradients to learn MLFN weights.      */
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

static TrainingSet *local_tptr ;   // These two statics pass class data
static MLFN *local_netptr ;        // to the local criterion routine
static int reg ;                   // Use regression for output weights?
static double *weights ;           // The weights being optimized are here
static int nvars ;                 // There are this many of them
static SingularValueDecomp *sptr ; // For 'regress'
static double *grad1, *grad2 ;     // Gradient work vectors


/*
   This evaluates the function.  If that value is less than 'limit',
   the gradient is also evaluated (if use_grad != 0).
   If the user pressed ESCape, the negative function value is returned.
*/

static double crit ( double *x , double limit , int use_grad , double *grad )
{
   double fval ;

   memcpy ( weights , x , nvars * sizeof(double) ) ;

   if (reg) {
      fval = local_netptr->regress ( local_tptr , sptr ) ;
      if ((fval < 0.0)  ||  user_pressed_escape ())
         return -fabs ( fval ) ;
      if (use_grad  &&  (fval < limit))    // Need gradient, fval redundant
         fval = local_netptr->gradient ( local_tptr , grad1 , grad2 , grad ) ;
      }

   else if (use_grad)
      fval = local_netptr->gradient ( local_tptr , grad1 , grad2 , grad ) ;

   else
      fval = local_netptr->trial_error ( local_tptr ) ;

   if ((fval < 0.0)  ||  user_pressed_escape ())
      return -fabs ( fval ) ;

   return fval ;
}

int MLFN::ssg (
   TrainingSet *tptr ,        // Training set to use
   struct LearnParams *lptr , // User's general learning parameters
   int use_grad               // SS if zero, else SSG
   )
{
   int i, itry, user_quit, improved, ntemps, niters, setback, n ;
   long seed ;
   double fval, bestfval, starttemp, stoptemp, fquit, *grad, *avg_grad ;
   double *x, *best, *work1, *work2 ;
   char msg[80] ;
   enum RandomDensity density ;
   struct AnnealParams *aptr ; // User's annealing parameters
                             
/*
   Get local copies of all annealing parameters
*/

   aptr = lptr->ap ;
   ntemps = aptr->ntempI ;
   niters = aptr->niterI ;
   setback = aptr->sbI ;
   starttemp = aptr->startI ;
   stoptemp = aptr->endI ;
   density = aptr->randomI ;

   if (! (ntemps * niters))
      return 0 ;

   fquit = lptr->quit_err ;

/*
   Allocate the singular value decomposition object for REGRESS.
   Note that there is no sense using regression if there are no hidden layers.
*/

   reg = nhid1 ;

   if (reg) {
      i = (domain == DOMAIN_COMPLEX)  ?  2 * tptr->ntrain : tptr->ntrain ;

      if (nhid2 == 0)         // One hidden layer
         n = nhid1_n ;
      else                    // Two hidden layers
         n = nhid2_n ;

      if (i < n) {
         write_progress ( "Too few training sets for regression." ) ;
         reg = 0 ;
         }
      else {
         MEMTEXT ( "SSG: new SingularValueDecomp" ) ;
         sptr = new SingularValueDecomp ( i , n , 1 ) ;

         if ((sptr == NULL)  || ! sptr->ok) {
            write_progress ( "Insufficient memory for regression." ) ;
            if (sptr != NULL)
               delete sptr ;
            reg = 0 ;
            sptr = NULL ;
            }
         }
      }

/*
   Initialize for 'crit' which is called from the annealing routines
*/

   if (reg)
      nvars = nhid ;
   else
      nvars = ntot ;

   weights = all_weights ;

/*
   Allocate scratch memory
*/

   MEMTEXT ( "SSG:: x, best, work1, work2" ) ;
   x = (double *) MALLOC ( nvars * sizeof(double) ) ;
   best = (double *) MALLOC ( nvars * sizeof(double) ) ;
   work1 = (double *) MALLOC ( nvars * sizeof(double) ) ;
   work2 = (double *) MALLOC ( nvars * sizeof(double) ) ;

   if ((x == NULL) || (best == NULL) || (work1 == NULL) ||  (work2 == NULL)) {
      if (x != NULL)
         FREE ( x ) ;
      if (best != NULL)
         FREE ( best ) ;
      if (work1 != NULL)
         FREE ( work1 ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      if (reg)
         delete sptr ;
      errtype = 0 ;
      return -1 ;
      }

/*
   Allocate gradient work memory.
   Grad1 is used for hidden layer 2 deltas in REAL model, and output
   activation partial derivatives and deltas in all COMPLEX models.
   Grad2 is output deltas in REAL model, error difference in COMPLEX models.
*/

   if (use_grad) {
      if (nhid2)       // Must be REAL model if this is true
         n = nhid2 ;
      else if (domain == DOMAIN_COMPLEX_INPUT)
         n = nhid1  ?  n_outputs * 2 + nhid1 * 2  :  n_outputs * 2 ;
      else if (domain == DOMAIN_COMPLEX_HIDDEN)
         n = n_outputs * 4  +  nhid1 * 4 ;
      else if (domain == DOMAIN_COMPLEX)
         n = nhid1  ?  n_outputs * 6  +  nhid1 * 4  :  n_outputs * 4 ;
      else
         n = 0 ;

      if (n) {
         MEMTEXT ( "SSG::grad1" ) ;
         grad1 = (double *) MALLOC ( n * sizeof(double) ) ;
         if (grad1 == NULL) {
            FREE ( x ) ;
            FREE ( best ) ;
            FREE ( work1 ) ;
            FREE ( work2 ) ;
            if (reg)
               delete sptr ;
            errtype = 0 ;
            return -1 ;
            }
         }
      else
         grad1 = NULL ;

      MEMTEXT ( "SSG::grad2, grad, avg_grad" ) ;
      grad2 = (double *) MALLOC ( nout_n * sizeof(double) ) ;
      grad = (double *) MALLOC ( ntot * sizeof(double) ) ; // 'gradient' use!
      avg_grad = (double *) MALLOC ( nvars * sizeof(double) ) ;

      if ((grad2 == NULL)  ||  (grad == NULL)  ||  (avg_grad == NULL)) {
         if (grad1 != NULL)
            FREE ( grad1 ) ;
         if (grad2 != NULL)
            FREE ( grad2 ) ;
         if (grad != NULL)
            FREE ( grad ) ;
         if (avg_grad != NULL)
            FREE ( avg_grad ) ;
         FREE ( x ) ;
         FREE ( best ) ;
         FREE ( work1 ) ;
         FREE ( work2 ) ;
         if (reg)
            delete sptr ;
         errtype = 0 ;
         return -1 ;
         }
      }
   else
      grad1 = grad2 = grad = avg_grad = NULL ;

   local_netptr = this ;
   local_tptr = tptr ;

/*
   If this is being used to initialize the weights, make sure that they are
   not identically zero.  Do this by setting bestfval huge so that
   SOMETHING is accepted later.
*/

   bestfval = 1.e30 ;
   i = ntot ;
   while (i--) {
      if (fabs(all_weights[i]) > 1.e-10) {
         bestfval = trial_error ( tptr ) ;
         break ;
         }
      }

/*
   This is the main loop.  If this is initial training of a new network,
   'weights' will be all zeros.  But we may be retraining a previously
   trained network, in which case 'weights' is important.
   Copy the weights to 'x', which is the parameter vector that will be
   optimized.  Also copy the weights to 'best', which will always hold
   the best weights so far.
   We need 'improved' we may use regression to find the output weights.
   If so and improvement was had, then regression must be used
   before returning the final weights.
*/

   memcpy ( x , weights , nvars * sizeof(double) ) ;    // May be retraining
   memcpy ( best , weights , nvars * sizeof(double) ) ; // Keep best here
   improved = 0 ;

   make_progress_window ( "Stochastic smoothing MLFN learning" ) ;

   for (itry=1 ; itry<=lptr->retries+1 ; itry++) {

      fval = ssg_core ( nvars , x , crit , 1.e30 , ntemps , niters ,
                        setback , starttemp , stoptemp ,
                        density , fquit , use_grad , work1 , work2 ,
                        grad , avg_grad , lptr->progress ) ;

      user_quit = (fval < 0.0) ;
      neterr = fabs ( fval ) ;  // Fval negative if user pressed ESCape

      if (neterr < bestfval) {
         bestfval = neterr ;
         memcpy ( best , x , nvars * sizeof(double) ) ;
         improved = 1 ;
         }

      sprintf ( msg , "Try %d  err=%lf  best=%lf", itry, neterr, bestfval ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      if (user_quit  ||  (neterr < fquit))
         break ;

      seed = flrand() / 2L - (long) (itry * 97) ; // Insure new seed
      sflrand ( seed ) ;
      for (i=0 ; i<nvars ; i++)  // Retry from zero origin
         x[i] = 0.0 ;
      }

/*
   If we are using regression and we improved,
   we only kept the best hidden weights.  Compute them now.
*/

   memcpy ( weights , best , nvars * sizeof(double) ) ;
   neterr = bestfval ;

   if (improved  &&  reg)
      neterr = regress ( tptr , sptr ) ; // regressed output weights

   sprintf ( msg , "Stochastic smoothing final error = %.6lf", neterr ) ;
   if (lptr->progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;


   if (reg) {
      MEMTEXT ( "SSG::delete sptr" ) ;
      delete sptr ;
      }

   MEMTEXT ( "SSG::x, best, work1, work" ) ;
   FREE ( x ) ;
   FREE ( best ) ;
   FREE ( work1 ) ;
   FREE ( work2 ) ;
   if (use_grad) {
      if (grad1 != NULL) {
         MEMTEXT ( "SSG::grad1" ) ;
         FREE ( grad1 ) ;
         }
      MEMTEXT ( "SSG::grad2, grad, avg_grad" ) ;
      FREE ( grad2 ) ;
      FREE ( grad ) ;
      FREE ( avg_grad ) ;
      }

   destroy_progress_window () ;

   if (user_quit)
      return 1 ;
   return 0 ;
}
