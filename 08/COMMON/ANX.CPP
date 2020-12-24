/******************************************************************************/
/*                                                                            */
/*  ANX - Learn by a pure anneal method                                       */
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
static SingularValueDecomp *sptr ;

static double crit ( double *x )
{
   memcpy ( weights , x , nvars * sizeof(double) ) ;

   if (reg)
      return local_netptr->regress ( local_tptr , sptr ) ;
   else
      return local_netptr->trial_error ( local_tptr ) ;
}


int MLFN::anx ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int i, n, itry, ntemps, niters, setback, user_quit, improved, seed ;
   int climb, reduction ;
   char msg[400] ;
   double fval, bestfval, starttemp, stoptemp, fquit, ratio ;
   double *x, *best, *work, *work2 ;
   enum RandomDensity density ;
   struct AnnealParams *aptr ; // User's annealing parameters

/*
   Get local copies of all annealing parameters
*/

   aptr = lptr->ap ;

// These are common to all methods
   ntemps = aptr->ntempI ;
   niters = aptr->niterI ;
   setback = aptr->sbI ;
   starttemp = aptr->startI ;
   stoptemp = aptr->endI ;
   density = aptr->randomI ;
// These are for ANNEAL2 only
   ratio = aptr->ratioI ;
   climb = aptr->climbI ;
   reduction = aptr->reducI ;


   if (! (ntemps * niters))
      return 0 ;

   fquit = lptr->quit_err ;

/*
   Allocate the singular value decomposition object for REGRESS.
   Do not use regression if there are no hidden layers or not MEAN SQUARE.
*/

   reg = nhid1  &&  (errtype == ERRTYPE_MSE) ;

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
         MEMTEXT ( "ANX: new SingularValueDecomp" ) ;
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

   MEMTEXT ( "ANX:: x, best, work" ) ;
   x = (double *) MALLOC ( nvars * sizeof(double) ) ;
   best = (double *) MALLOC ( nvars * sizeof(double) ) ;
   work = (double *) MALLOC ( nvars * sizeof(double) ) ;
   if (lptr->method == METHOD_AN2)
      work2 = (double *) MALLOC ( nvars * sizeof(double) ) ;
   else
      work2 = NULL ;

   if ((x == NULL) || (work == NULL) || (best == NULL) ||
       ((lptr->method == METHOD_AN2) && (work2 == NULL))) {
      if (x != NULL)
         FREE ( x ) ;
      if (best != NULL)
         FREE ( best ) ;
      if (work != NULL)
         FREE ( work ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      if (reg)
         delete sptr ;
      errtype = 0 ;
      return -1 ;
      }

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
   The flag 'improved' is needed because we may be using regression to find
   the output weights.  If we improved, which certainly is the hope,
   then regression must be used one last time before returning to the caller.
*/

   memcpy ( x , weights , nvars * sizeof(double) ) ;    // May be retraining
   memcpy ( best , weights , nvars * sizeof(double) ) ; // Keep best here
   improved = 0 ;

   if (lptr->method == METHOD_AN1)
      make_progress_window ( "AN1 MLFN learning" ) ;
   if (lptr->method == METHOD_AN2)
      make_progress_window ( "AN2 MLFN learning" ) ;

   for (itry=1 ; itry<=lptr->retries+1 ; itry++) {

      if (lptr->method == METHOD_AN1)
         fval = anneal1 ( nvars , x , work , crit , 1.e30 , itry , ntemps ,
                          niters , setback , starttemp , stoptemp ,
                          density , fquit , lptr->progress ) ;
      else if (lptr->method == METHOD_AN2)
         fval = anneal2 ( nvars , x , work , work2 , crit , 1.e30 , ntemps ,
                          niters , setback , starttemp , stoptemp , density ,
                          ratio , climb , reduction , fquit , lptr->progress) ;

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

      seed = flrand() / 2 + (itry * 97) ; // Insure new seed for anneal
      sflrand ( seed ) ;
      for (i=0 ; i<nvars ; i++)  // Retry from zero origin
         x[i] = 0.0 ;
      }


/*
   We must be wary of an easy mistake if we are using regression.
   If there was improvement, we only recorded (via the random seed) the
   optimal hidden weights, not the output weights.  So we must call regress
   now to compute the output weights.
*/

   memcpy ( weights , best , nvars * sizeof(double) ) ;
   neterr = bestfval ;

   if (improved  &&  reg)
      neterr = regress ( tptr , sptr ) ; // regressed output weights

   if (reg) {
      MEMTEXT ( "ANX: delete SingularValueDecomp" ) ;
      delete sptr ;
      }

   sprintf ( msg , "ANX final error = %.6lf", neterr ) ;
   if (lptr->progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;

   MEMTEXT ( "ANX::x, work, best" ) ;
   FREE ( x ) ;
   FREE ( best ) ;
   FREE ( work ) ;
   if (work2 != NULL)
      FREE ( work2 ) ;

   destroy_progress_window () ;

   if (user_quit)
      return 1 ;
   return 0 ;
}
