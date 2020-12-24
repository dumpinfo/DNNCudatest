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
/*  ANX_DD - Learn by hybrid of anneal method plus direct descent             */
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

void LayerNet::anx_dd ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int itry, n_escape, n_retry, bad_count, new_record, refined ;
   long seed ;
   double err, prev_err, best_err, start_of_loop_error, best_inner_error ;
   double initial_accuracy, final_accuracy ;
   char msg[80] ;
   LayerNet *worknet, *worknet2, *bestnet ;

   n_escape = n_retry = 0 ;

/*
   Allocate scratch memory
*/

   MEMTEXT ( "ANX_DD::learn new worknet, bestnet" ) ;
   worknet = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                            nout , 0 , 0 ) ;
   bestnet = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                            nout , 0 , 1 ) ;

   if ((worknet == NULL)  ||  (! worknet->ok)
    || (bestnet == NULL)  ||  (! bestnet->ok)) {
      memory_message ( "to learn" ) ;
      if (worknet != NULL)
         delete worknet ;
      if (bestnet != NULL)
         delete bestnet ;
      errtype = 0 ;
      return ;
      }

   if ((lptr->method == METHOD_AN2_CJ)  ||  (lptr->method == METHOD_AN2_LM)) {
      worknet2 = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                                nout , 0 , 0 ) ;
      if ((worknet2 == NULL)  ||  (! worknet2->ok)) {
         if (worknet2 != NULL)
            delete worknet2 ;
         delete worknet ;
         delete bestnet ;
         memory_message ( "to learn" ) ;
         errtype = 0 ;
         return ;
         }
      }
   else
      worknet2 = NULL ;

/*
   Start by annealing around the starting weights.  These will be zero if the
   net was just created.  If it was restored or partially trained already,
   they will be meaningful.  Anneal1 guarantees that it will not return all
   zero weights if there is at least one hidden layer, even if that means
   that the error exceeds the amount that could be attained by all zeros.
*/

   best_err = best_inner_error = 1.e30 ;
   if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN1_LM))
      anneal1 ( tptr , lptr , worknet , 1 , 0 ) ;
   else if ((lptr->method == METHOD_AN2_CJ) || (lptr->method == METHOD_AN2_LM))
      anneal2 ( tptr , lptr , worknet , worknet2 , 1 ) ;

/*
   Do direct descent optimization, finding local minimum.
   Then anneal to break out of it.  If successful, loop back up to
   do direct descent again.  Otherwise restart totally random.
*/

   bad_count = 0 ;         // Handles flat local mins
   refined = 0 ;           // Did we ever refine to high resolution?  Not yet.
   new_record = 0 ;        // Refine every time a new inner error record set
   initial_accuracy = pow ( 10.0 , -lptr->cj_acc ) ;
   final_accuracy = initial_accuracy * pow ( 10.0 , -lptr->cj_refine ) ;

   for (itry=1 ; ; itry++) {

      if (neterr < best_err) {   // Keep track of best
         copy_weights ( bestnet , this ) ;
         best_err = neterr ;
         }

      sprintf ( msg , "Try %d  (best=%lf):", itry, best_err ) ;
      normal_message ( msg ) ;

      if (neterr <= lptr->quit_err)
         break ;

      start_of_loop_error = neterr ;
      if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN2_CJ))
         err = conjgrad ( tptr , 32767 , initial_accuracy ,
                          lptr->quit_err , lptr->cj_progress ) ;
      else if ((lptr->method==METHOD_AN1_LM) || (lptr->method==METHOD_AN2_LM))
         err = lev_marq ( tptr , 32767 , initial_accuracy ,
                          lptr->quit_err , lptr->cj_progress ) ;
      neterr = fabs ( err ) ; // err<0 if user pressed ESCape

      sprintf ( msg , "  Gradient err=%lf", neterr ) ;
      progress_message ( msg ) ;

      if (neterr < best_err) {   // Keep track of best
         copy_weights ( bestnet , this ) ;
         best_err = neterr ;
         }

      if (err <= lptr->quit_err) { // err<0 if user pressed ESCape
         if (err < -1.e29)         // or insufficient memory
            printf ( "\nInsufficient memory for gradient learning." ) ;
         break ;
         }

      seed = flrand() - (long) (itry * 97) ;   // Insure new seed for anneal
      sflrand ( seed ) ;

      prev_err = neterr ;  // So we can see if anneal helped

      if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN1_LM))
         anneal1 ( tptr , lptr , worknet , 0 , itry ) ;
      else if ((lptr->method==METHOD_AN2_CJ) || (lptr->method==METHOD_AN2_LM))
         anneal2 ( tptr , lptr , worknet , worknet2 , 0 ) ;

      sprintf ( msg , "  Anneal err=%lf", neterr ) ;
      progress_message ( msg ) ;

      if (neterr < best_err) {  // Keep track of best
         copy_weights ( bestnet , this ) ;
         best_err = neterr ;
         }

      if (best_err <= lptr->quit_err)
         break ;

      if (neterr < best_inner_error) {  // Keep track of best inner for refine
         best_inner_error = neterr ;
         new_record = 1 ;               // Tells us to refine
         }

      if ((prev_err - neterr) > 1.e-7) { // Did we break out of local min?
         if ((start_of_loop_error - neterr) < 1.e-3)
            ++bad_count ;  // Avoid many unprofitable iters
         else
            bad_count = 0 ;
         if (bad_count < 4) {
            ++n_escape ;          // Count escapes from local min
            continue ;            // Escaped, so gradient learn again
            }
         }

/*
   After first few tries, and after each inprovement thereafter, refine
   to high resolution
*/

      if ((itry-n_escape >= lptr->cj_pretries)  &&  (new_record || ! refined)) {
         if (! refined) {   // If refining the best of the pretries
            copy_weights ( this , bestnet ) ;  // Get that net
            neterr = best_err ;
            }
         refined = 1 ;     // Only force refine once
         new_record = 0 ;  // Reset new inner error record flag
         progress_message ( "  REFINING" ) ;
         if ((lptr->method == METHOD_AN1_CJ) || (lptr->method == METHOD_AN2_CJ))
            err = conjgrad ( tptr , 0 , final_accuracy ,
                             lptr->quit_err , lptr->cj_progress ) ;
         else if ((lptr->method==METHOD_AN1_LM)|| (lptr->method==METHOD_AN2_LM))
            err = lev_marq ( tptr , 0 , final_accuracy ,
                             lptr->quit_err , lptr->cj_progress ) ;
         neterr = fabs ( err ) ; // err<0 if user pressed ESCape
         sprintf ( msg , "  Attained err=%lf", neterr ) ;
         progress_message ( msg ) ;
         if (neterr < best_err) {  // Keep track of best
            copy_weights ( bestnet , this ) ;
            best_err = neterr ;
            }
         }

      if (++n_retry > lptr->retries)
         break ;

      progress_message ( "  RESTART" ) ;
      zero_weights () ;  // Failed to break out, so retry random
      seed = flrand() - (long) (itry * 773) ;   // Insure new seed for anneal
      sflrand ( seed ) ;
      if ((lptr->method == METHOD_AN1_CJ)  ||  (lptr->method == METHOD_AN1_LM))
         anneal1 ( tptr , lptr , worknet , 1 , itry ) ;
      else if ((lptr->method==METHOD_AN2_CJ) || (lptr->method==METHOD_AN2_LM))
         anneal2 ( tptr , lptr , worknet , worknet2 , 1 ) ;
      }

FINISH:
   copy_weights ( this , bestnet ) ;
   neterr = best_err ;
   MEMTEXT ( "AN1_DD::learn delete worknet, bestnet" ) ;
   delete worknet ;
   delete bestnet ;
   sprintf ( msg , "%d successful escapes, %d retries", n_escape, n_retry ) ;
   normal_message ( msg ) ;

   return ;
}
