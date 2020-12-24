/******************************************************************************/
/*                                                                            */
/*  ANX_DD - Learn by hybrid of anneal method plus direct descent             */
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

#define DEBUG_GRAD 0
#define DELTA 0.0000001

static TrainingSet *local_tptr ;   // These two statics pass class data
static MLFN *local_netptr ;        // to the local criterion routine
static double *weights ;           // The weights being optimized are here
static int reg ;                   // Use regression for output weights?
static int nvars ;                 // There are this many of them
static double *grad1, *grad2 ;     // Gradient work vectors
static SingularValueDecomp *sptr, *lmsptr ;


/*
   These evaluate the function.
   The first, 'acrit', is for annealing and uses regression.
   The second, 'dcrit', is for conjugate gradients.
   If 'find_grad' is nonzero, the gradient is also evaluated.
   The third, 'lcrit', is used for Levenberg-Marquardt learning.
*/

static double acrit ( double *x )
{
   memcpy ( weights , x , nvars * sizeof(double) ) ;

   if (reg)
      return local_netptr->regress ( local_tptr , sptr ) ;
   else
      return local_netptr->trial_error ( local_tptr ) ;
}


static double dcrit ( double *x , int find_grad , double *grad )
{
   double retval ;
   memcpy ( local_netptr->all_weights , x, local_netptr->ntot * sizeof(double));

#if DEBUG_GRAD
   int i ;
   double fval, f0, f1, deriv, dot, len1, len2 ;

   if (find_grad) {
      len1 = len2 = dot = 0.0 ;
      fval = local_netptr->gradient ( local_tptr , grad1 , grad2 , grad ) ;
      printf ( "\nCHK:" ) ;
      for (i=0 ; i<local_netptr->ntot ; i++) {
         local_netptr->all_weights[i] += DELTA ;
         f0 = local_netptr->trial_error ( local_tptr ) ;
         local_netptr->all_weights[i] -= 2.0 * DELTA ;
         f1 = local_netptr->trial_error ( local_tptr ) ;
         local_netptr->all_weights[i] += DELTA ;
         deriv = (f1 - f0) / (2.0 * DELTA) ;
         len1 += grad[i] * grad[i] ;
         len2 += deriv * deriv ;
         dot += grad[i] * deriv ;
#if DEBUG_GRAD > 1
         printf ( " (%lf %lf)", 100.0 * deriv, 100.0 * grad[i] ) ;
#endif
         }
      printf ( "  DOT=%lf", dot / sqrt ( len1 * len2 ) ) ;
      return fval ;
      }
   else
      return local_netptr->trial_error ( local_tptr ) ;
#endif
   if (find_grad) {
      retval = local_netptr->gradient ( local_tptr , grad1 , grad2 , grad ) ;
      return retval ;
      }
   else {
      retval = local_netptr->trial_error ( local_tptr ) ;
      return retval ;
      }
}

static double lcrit ( double *x , double *hessian , double *grad )
{
   memcpy ( local_netptr->all_weights , x, local_netptr->ntot * sizeof(double));
   return local_netptr->lm_core ( local_tptr , grad1 , grad2 , hessian , grad );
}

int MLFN::anx_dd ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int i, itry, n, n_escape, n_retry, bad_count, new_record, refined, user_quit;
   int n3, n4 ;
   int ntempI, niterI, sbI, climbI, reductionI ;
   int ntempE, niterE, sbE, climbE, reductionE ;
   long seed ;
   double fval, prev_err, bestfval, start_of_loop_error, best_inner_error ;
   double initial_accuracy, final_accuracy, fquit ;
   double starttempI, endtempI, ratioI, starttempE, endtempE, ratioE ;
   double *x, *best, *work1, *work2, *work3, *work4 ;
   char msg[400] ;
   enum RandomDensity densityI, densityE ;
   struct AnnealParams *aptr ; // User's annealing parameters

/*
   Get local copies of all annealing parameters
*/

   aptr = lptr->ap ;

// These are common to all methods
   ntempI = aptr->ntempI ;
   ntempE = aptr->ntempE ;
   niterI = aptr->niterI ;
   niterE = aptr->niterE ;
   sbI = aptr->sbI ;
   sbE = aptr->sbE ;
   starttempI = aptr->startI ;
   starttempE = aptr->startE ;
   endtempI = aptr->endI ;
   endtempE = aptr->endE ;
   densityI = aptr->randomI ;
   densityE = aptr->randomE ;
// These are for ANNEAL2 only
   ratioI = aptr->ratioI ;
   ratioE = aptr->ratioE ;
   climbI = aptr->climbI ;
   climbE = aptr->climbE ;
   reductionI = aptr->reducI ;
   reductionE = aptr->reducE ;

   if (! (ntempE * niterE * ntempI * niterI))
      return 0 ;

   fquit = lptr->quit_err ;

/*
   Allocate the singular value decomposition object for REGRESS.
   Note that there is no sense using regression if there are no hidden layers.
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
         MEMTEXT ( "ANX_DD: new SingularValueDecomp for regress" ) ;
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
   Allocate the singular value decomposition object for Levenberg-Marquardt.
*/

   if ((lptr->method == METHOD_AN1_LM) || (lptr->method == METHOD_AN2_LM)) {
      MEMTEXT ( "ANX_DD: new SingularValueDecomp for LM" ) ;
      lmsptr = new SingularValueDecomp ( ntot , ntot , 0 ) ;
      if ((lmsptr == NULL)  || ! lmsptr->ok) {
         if (lmsptr != NULL)
            delete lmsptr ;
         if (reg)
            delete sptr ;
         errtype = 0 ;
         return -1 ;
         }
      }
   else
      lmsptr = NULL ;

/*
   Initialize for 'acrit' which is called from the annealing routines
*/

   if (reg)
      nvars = nhid ;
   else
      nvars = ntot ;

   weights = all_weights ;

/*
   Allocate scratch memory
*/

   if (nhid2)       // Must be REAL model if this is true
      n = nhid2 ;
   else if (domain == DOMAIN_COMPLEX_INPUT)
      n = nhid1  ?  n_outputs * 2 + nhid1 * 2  :  n_outputs * 2 ;
   else if (domain == DOMAIN_COMPLEX_HIDDEN)
      n = n_outputs * 4  +  nhid1 * 4 ;
   else if (domain == DOMAIN_COMPLEX)
      n = nhid1  ?  n_outputs * 6  +  nhid1 * 4  :  n_outputs * 4 ;
   else
      n = 1 ;  // Not used, but allocating something simplifies bookkeeping

   MEMTEXT ( "ANX_DD: x, best, 4 works, 2 grads" ) ;
   x = (double *) MALLOC ( ntot * sizeof(double) ) ;
   best = (double *) MALLOC ( ntot * sizeof(double) ) ;
   work1 = (double *) MALLOC ( ntot * sizeof(double) ) ;
   work2 = (double *) MALLOC ( ntot * sizeof(double) ) ;
   if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN2_CJ)) {
      grad2 = (double *) MALLOC ( nout_n * sizeof(double)) ;
      n3 = n4 = ntot ;
      }
   else if ((lptr->method == METHOD_AN1_LM) || (lptr->method == METHOD_AN2_LM)){
      grad2 = (double *) MALLOC ( ntot * sizeof(double)) ;
      n3 = ntot * ntot ;
      n4 = 1 ; // Not used, but allocating something simplifies bookkeeping
      }
   work3 = (double *) MALLOC ( n3 * sizeof(double) ) ;
   work4 = (double *) MALLOC ( n4 * sizeof(double) ) ;
   grad1 = (double *) MALLOC ( n * sizeof(double) ) ;

   if ((x == NULL) || (best == NULL) || (work1 == NULL) || (work2 == NULL)
    || (work3 == NULL) || (work4 == NULL) || (grad1 == NULL) || (grad2 ==NULL)){
      if (x != NULL)
         FREE ( x ) ;
      if (best != NULL)
         FREE ( best ) ;
      if (work1 != NULL)
         FREE ( work1 ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      if (work3 != NULL)
         FREE ( work3 ) ;
      if (work4 != NULL)
         FREE ( work4 ) ;
      if (grad1 != NULL)
         FREE ( grad1 ) ;
      if (grad2 != NULL)
         FREE ( grad2 ) ;
      if (reg)
         delete sptr ;
      if (lmsptr)
         delete lmsptr ;
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
   Start by annealing around the starting weights.  These will be zero if the
   net was just created.  If it was restored or partially trained already,
   they will be meaningful.
*/

   memcpy ( x , all_weights , ntot * sizeof(double) ) ;    // May be retraining
   memcpy ( best , all_weights , ntot * sizeof(double) ) ;

   if (lptr->method == METHOD_AN1_CJ)
      make_progress_window ( "AN1_CJ MLFN learning" ) ;
   else if (lptr->method == METHOD_AN2_CJ)
      make_progress_window ( "AN2_CJ MLFN learning" ) ;
   else if (lptr->method == METHOD_AN1_LM)
      make_progress_window ( "AN1_LM MLFN learning" ) ;
   else if (lptr->method == METHOD_AN2_LM)
      make_progress_window ( "AN2_LM MLFN learning" ) ;

   if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN1_LM))
      fval = anneal1 ( nvars , x , work1 , acrit , bestfval , 99 , ntempI ,
             niterI , sbI , starttempI , endtempI ,
             densityI , fquit , lptr->progress ) ;
   else if ((lptr->method == METHOD_AN2_CJ) || (lptr->method == METHOD_AN2_LM))
      fval = anneal2 ( nvars , x , work1 , work2 , acrit , bestfval , ntempI ,
                       niterI , sbI , starttempI , endtempI , densityI ,
                       ratioI , climbI , reductionI, fquit , lptr->progress) ;

   memcpy ( all_weights , x , ntot * sizeof(double) ) ;
   memcpy ( best , all_weights , ntot * sizeof(double) ) ;

   user_quit = (fval < 0.0) ;

   if (reg) {
      neterr = regress ( tptr , sptr ) ; // regressed output weights
      memcpy ( x , all_weights , ntot * sizeof(double) ) ;
      }
   else 
      neterr = fabs ( fval ) ;  // Fval negative if user pressed ESCape

   if (user_quit) {
      memcpy ( best , all_weights , ntot * sizeof(double) ) ;
      bestfval = neterr ;
      goto FINISH ;
      }

   best_inner_error = 1.e30 ;

/*
   Do direct descent optimization, finding local minimum.
   Then anneal to break out of it.  If successful, loop back up to
   do direct descent again.  Otherwise restart totally random.
*/

   bad_count = 0 ;         // Handles flat local mins
   refined = 0 ;           // Did we ever refine to high resolution?  Not yet.
   new_record = 0 ;        // Refine every time a new inner error record set
   n_escape = n_retry = 0 ;
   initial_accuracy = pow ( 10.0 , -lptr->acc ) ;
   final_accuracy = initial_accuracy * pow ( 10.0 , -lptr->refine ) ;

   for (itry=1 ; ; itry++) {

      if (neterr < bestfval) {
         bestfval = neterr ;
         memcpy ( best , x , ntot * sizeof(double) ) ;
         }

      sprintf ( msg , "Try %d  (best=%lf  escapes=%d):",
                itry, bestfval, n_escape ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      user_quit |= user_pressed_escape () ;

      if (user_quit  ||  (neterr <= lptr->quit_err))
         break ;

      start_of_loop_error = neterr ;
      if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN2_CJ))
         fval = conjgrad ( 32767 , lptr->quit_err , initial_accuracy ,
                           dcrit , ntot , x , neterr , work1 , work2 , work3 ,
                           work4 , lptr->progress ) ;
      else if ((lptr->method==METHOD_AN1_LM) || (lptr->method==METHOD_AN2_LM))
         fval = lev_marq ( 0 , lptr->quit_err , initial_accuracy , lcrit ,
                           ntot , x , lmsptr , work1 , work2 , work3 ,
                           lptr->progress ) ;

      user_quit = (fval < 0.0) ;
      neterr = fabs ( fval ) ;    // fval<0 if user pressed ESCape

      sprintf ( msg , "  Gradient err=%lf", neterr ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;

      if (neterr < bestfval) {   // Keep track of best
         memcpy ( best , x , ntot * sizeof(double) ) ;
         bestfval = neterr ;
         }

      if (neterr <= lptr->quit_err)
         break ;

      if (user_quit)
         break ;

      seed = flrand() / 2 + (itry * 97) ;  // Insure new seed for anneal
      sflrand ( seed ) ;

      prev_err = neterr ;  // So we can see if anneal helped

      if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN1_LM))
         fval = anneal1 ( nvars , x , work1 , acrit , neterr , itry , ntempE ,
                niterE , sbE , starttempE , endtempE ,
                densityE , fquit , lptr->progress ) ;
      else if ((lptr->method == METHOD_AN2_CJ) || (lptr->method==METHOD_AN2_LM))
         fval = anneal2 ( nvars , x , work1 , work2 , acrit , neterr , ntempE ,
                          niterE , sbE , starttempE , endtempE , densityE ,
                          ratioE , climbE , reductionE, fquit , lptr->progress ) ;

      user_quit = (fval < 0.0) ;

      if (reg) {
         memcpy ( all_weights , x , ntot * sizeof(double) ) ;
         neterr = regress ( tptr , sptr ) ; // regressed output weights
         memcpy ( x , all_weights , ntot * sizeof(double) ) ;
         }
      else 
         neterr = fabs ( fval ) ;  // Fval negative if user pressed ESCape


      if (neterr < bestfval) {
         bestfval = neterr ;
         memcpy ( best , x , ntot * sizeof(double) ) ;
         }

      user_quit |= user_pressed_escape () ;
      if (user_quit)
         break ;

      if (bestfval <= lptr->quit_err)
         break ;

      if (neterr < best_inner_error) {  // Keep track of best inner for refine
         best_inner_error = neterr ;
         new_record = 1 ;               // Tells us to refine
         }

      if ((prev_err - neterr) > 1.e-7) { // Did we break out of local min?
         if ((start_of_loop_error - neterr) < 0.01)
            ++bad_count ;  // Avoid many unprofitable iters
         else
            bad_count = 0 ;
         if (bad_count < 4) {
            ++n_escape ;          // Count escapes from local min
            sprintf ( msg , "ESCAPED Try %d  (best=%lf  escapes=%d):",
                      itry, bestfval, n_escape ) ;
            if (lptr->progress)
               write_progress ( msg ) ;
            else 
               write_non_progress ( msg ) ;
            continue ;            // Escaped, so gradient learn again
            }
         }

/*
   After first few tries, and after each improvement thereafter, refine
   to high resolution
*/

      if ((itry-n_escape >= lptr->pretries)  &&  (new_record || ! refined)) {
         if (! refined) {   // If refining the best of the pretries
            memcpy ( x , best , ntot * sizeof(double) ) ;
            neterr = bestfval ;
            }
         refined = 1 ;     // Only force refine once
         new_record = 0 ;  // Reset new inner error record flag
         sprintf ( msg , "REFINING Try %d  (best=%lf  escapes=%d):",
                   itry, bestfval, n_escape ) ;
         if (lptr->progress)
            write_progress ( msg ) ;
         else 
            write_non_progress ( msg ) ;
         if ((lptr->method == METHOD_AN1_CJ) || (lptr->method == METHOD_AN2_CJ))
            fval = conjgrad ( 32767 , lptr->quit_err , final_accuracy ,
                              dcrit , ntot , x , neterr , work1 , work2 , work3,
                              work4 , lptr->progress ) ;
         else if ((lptr->method==METHOD_AN1_LM)|| (lptr->method==METHOD_AN2_LM))
            fval = lev_marq ( 0 , lptr->quit_err , final_accuracy , lcrit ,
                              ntot , x , lmsptr , work1 , work2 , work3 ,
                              lptr->progress ) ;
         user_quit = (fval < 0.0) ;
         neterr = fabs ( fval ) ; // err<0 if user pressed ESCape
         if (neterr < bestfval) {  // Keep track of best
            bestfval = neterr ;
            memcpy ( best , x , ntot * sizeof(double) ) ;
            }
         sprintf ( msg , "  Refined err=%lf", neterr ) ;
         if (lptr->progress)
            write_progress ( msg ) ;
         else 
            write_non_progress ( msg ) ;
         if (user_quit)
            break ;
         }

      if (++n_retry > lptr->retries)
         break ;

      for (i=0 ; i<ntot ; i++)  // Retry from zero origin
         x[i] = 0.0 ;
      if (lptr->progress)
         write_progress ( "  RESET" ) ;
      else 
         write_non_progress ( "  RESET" ) ;
      seed = flrand() - (long) (itry * 773) ;   // Insure new seed for anneal
      sflrand ( seed ) ;
      if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN1_LM))
         fval = anneal1 ( nvars , x , work1 , acrit , 1.e30 , 99 , ntempI ,
                niterI , sbI , starttempI , endtempI ,
                densityI , fquit , lptr->progress ) ;
      else if ((lptr->method == METHOD_AN2_CJ) || (lptr->method == METHOD_AN2_LM))
         fval = anneal2 ( nvars , x , work1 , work2 , acrit , 1.e30 , ntempI ,
                       niterI , sbI , starttempI , endtempI , densityI ,
                       ratioI , climbI , reductionI, fquit , lptr->progress ) ;

      user_quit = (fval < 0.0) ;

      if (reg) {
         memcpy ( all_weights , x , ntot * sizeof(double) ) ;
         neterr = regress ( tptr , sptr ) ; // regressed output weights
         memcpy ( x , all_weights , ntot * sizeof(double) ) ;
         }
      else 
         neterr = fabs ( fval ) ;  // Fval negative if user pressed ESCape

      }

FINISH:
   memcpy ( all_weights , best , ntot * sizeof(double) ) ;
   neterr = bestfval ;
   if (reg &&  outlin) // Final (usually tiny) tweaking of weights
      neterr = regress ( tptr , sptr ) ; // regressed output weights
   MEMTEXT ( "ANX_DD: works (7)" ) ;
   FREE ( x ) ;
   FREE ( best ) ;
   FREE ( work1 ) ;
   FREE ( work2 ) ;
   FREE ( work3 ) ;
   FREE ( work4 ) ;
   FREE ( grad2 ) ;
   if (grad1 != NULL) {
      MEMTEXT ( "ANX_DD: grad1" ) ;
      FREE ( grad1 ) ;
      }
   if (reg) {
      MEMTEXT ( "ANX_DD: delete sptr" ) ;
      delete sptr ;
      }
   if (lmsptr) {
      MEMTEXT ( "ANX_DD: delete lmsptr" ) ;
      delete lmsptr ;
      }

   if (errtype) {
      sprintf ( msg , "Final error = %.6lf", neterr ) ;
      if (lptr->progress)
         write_progress ( msg ) ;
      else 
         write_non_progress ( msg ) ;
      }

   destroy_progress_window () ;

   if (user_quit)
      return 1 ;
   return 0 ;
}
