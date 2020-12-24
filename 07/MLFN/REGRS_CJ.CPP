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
/*  REGRS_CJ - Learn by hybrid of regression plus conjugate gradient          */
/*                                                                            */
/*  This method is valid only if there is no hidden layer.                    */
/*  If the output is linear and MSE error used, this is direct: call regress. */
/*  Otherwise we call regress to get the starting weights, then call conjgrad */
/*  to optimize.  We may call anneal1 to break out of a local minimum, but    */
/*  there is no point to an outermost loop as in AN1_CJ.                      */
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

void LayerNet::regrs_cj ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int i, itry, bad_count ;
   long seed ;
   double err, prev_err, best_err, start_of_loop_error ;
   char msg[80] ;
   LayerNet *worknet, *bestnet ;
   SingularValueDecomp *sptr ;

/*
   Allocate scratch memory
*/

   i = (model == NETMOD_COMPLEX)  ?  2 * tptr->ntrain : tptr->ntrain ;
   if (i < nin_n) {
      error_message ( "Too few training sets for regression." ) ;
      errtype = 0 ;
      return ;
      }

   MEMTEXT ( "REGRS_CJ::learn new SingularValueDecomp, worknet, bestnet" ) ;

   sptr = new SingularValueDecomp ( i , nin_n , 1 ) ;
   if ((sptr == NULL)  || ! sptr->ok) {
      memory_message ( "for regression." ) ;
      if (sptr != NULL)
         delete sptr ;
      errtype = 0 ;
      return ;
      }

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
      delete sptr ;
      errtype = 0 ;
      return ;
      }

/*
   The starting weights are directly computed by regression.
   If the outputs are linear and we are using MSE, this is all we need.
*/

   best_err = regress ( tptr , sptr ) ;
   copy_weights ( bestnet , this ) ;
   if (outlin  &&  (errtype == ERRTYPE_MSE)) {
      sprintf ( msg , " err=%lf", best_err ) ;
      normal_message ( msg ) ;
      goto FINISH ;
      }

/*
   Do conjugate gradient optimization, finding local minimum.
   Then anneal to break out of it.  If successful, loop back up to
   do conjugate gradient again.  Otherwise we are done.
*/

   bad_count = 0 ;           // Handles flat local mins

   for (itry=1 ; ; itry++) {

      sprintf ( msg , "Try %d  (best=%lf):", itry, best_err ) ;
      normal_message ( msg ) ;

      if (neterr <= lptr->quit_err)
         break ;

      start_of_loop_error = neterr ;
      err = conjgrad ( tptr, 32767, pow (10.0, -(lptr->cj_refine+lptr->cj_acc)),
                       lptr->quit_err, lptr->cj_progress);
      neterr = fabs ( err ) ; // err<0 if user pressed ESCape

      sprintf ( msg , "  Gradient err=%lf", neterr ) ;
      progress_message ( msg ) ;

      if (neterr < best_err) {   // Keep track of best
         copy_weights ( bestnet , this ) ;
         best_err = neterr ;
         }

      if (err <= lptr->quit_err) // err<0 if user pressed ESCape
         break ;

      seed = flrand() - (long) (itry * 97) ;   // Insure new seed for anneal
      sflrand ( seed ) ;

      prev_err = neterr ;  // So we can see if anneal helped
      anneal1 ( tptr , lptr , worknet , 0 ) ;

      sprintf ( msg , "  Anneal err=%lf", neterr ) ;
      progress_message ( msg ) ;

      if (neterr < best_err) {  // Keep track of best
         copy_weights ( bestnet , this ) ;
         best_err = neterr ;
         }

      if (best_err <= lptr->quit_err)
         break ;

      if (neterr < prev_err) { // Did we break out of local min?
         if ((start_of_loop_error - neterr) < 1.e-3)
            ++bad_count ;  // Avoid many unprofitable iters
         else
            bad_count = 0 ;
         if (bad_count < 4)
            continue ;            // Escaped, so gradient learn again
         }

      break ;
      }

FINISH:
   copy_weights ( this , bestnet ) ;
   neterr = best_err ;
   MEMTEXT ( "REGRS_CJ::learn delete SingularValueDecomp, worknet, bestnet" ) ;
   delete sptr ;
   delete worknet ;
   delete bestnet ;

   return ;
}
