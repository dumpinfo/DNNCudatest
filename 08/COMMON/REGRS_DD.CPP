/******************************************************************************/
/*                                                                            */
/*  REGRS_DD - Learn by hybrid of regression plus direct descent              */
/*                                                                            */
/*  This method is valid only if there is no hidden layer.                    */
/*  If the output is linear and MSE error used, this is direct: call regress. */
/*  Otherwise we call regress to get the starting weights, then call conjgrad */
/*  to optimize.                                                              */
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
static double *grad1, *grad2 ;     // Gradient work vectors
static SingularValueDecomp *sptr ;


/*
   These evaluate the function.
   The first, 'dcrit', is for conjugate gradients.
   If 'find_grad' is nonzero, the gradient is also evaluated.
   The third, 'lcrit', is used for Levenberg-Marquardt learning.
*/

static double dcrit ( double *x , int find_grad , double *grad )
{
   memcpy ( local_netptr->all_weights , x, local_netptr->ntot * sizeof(double));

   if (find_grad)
      return local_netptr->gradient ( local_tptr , grad1 , grad2 , grad ) ;
   else
      return local_netptr->trial_error ( local_tptr ) ;
}

static double lcrit ( double *x , double *hessian , double *grad )
{
   memcpy ( local_netptr->all_weights , x, local_netptr->ntot * sizeof(double));
   return local_netptr->lm_core ( local_tptr , grad1 , grad2 , hessian , grad );
}

int MLFN::regrs_dd ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int i, n3, n4, n_grad1, user_quit;
   double fval, final_accuracy ;
   double *x, *work1, *work2, *work3, *work4 ;
   char msg[80] ;

/*
   Allocate the singular value decomposition object for REGRESS.
*/

   i = (domain == DOMAIN_COMPLEX)  ?  2 * tptr->ntrain : tptr->ntrain ;

   if (i < nin_n) {
      write_progress ( "Too few training sets for regression." ) ;
      errtype = 0 ;
      return 1 ;
      }

   MEMTEXT ( "REGRS_DD: new SingularValueDecomp for regress" ) ;
   sptr = new SingularValueDecomp ( i , nin_n , 1 ) ;

   if ((sptr == NULL)  || ! sptr->ok) {
      if (sptr != NULL)
         delete sptr ;
      errtype = 0 ;
      return -1 ;
      }

/*
   The starting weights are directly computed by regression.
   If the outputs are linear and we are using MSE, this is all we need.
*/

   user_quit = 0 ;
   make_progress_window ( "REGRESSION initialized MLFN learning" ) ;

   neterr = regress ( tptr , sptr ) ;

   MEMTEXT ( "REGRS_DD: delete SingularValueDecomp for regress" ) ;
   delete sptr ;

   sprintf ( msg , "Regression err = %lf", neterr ) ;
   if (lptr->progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;

   if ((outlin  &&  (errtype == ERRTYPE_MSE))  ||  (neterr <= lptr->quit_err))
      goto FINISH ;

   user_quit = user_pressed_escape () ;
   if (user_quit)
      goto FINISH ;


/*
   Allocate the singular value decomposition object for Levenberg-Marquardt.
*/

   if (lptr->method == METHOD_REGRS_LM) {
      MEMTEXT ( "REGRS_DD: new SingularValueDecomp for LM" ) ;
      sptr = new SingularValueDecomp ( ntot , ntot , 0 ) ;
      if ((sptr == NULL)  || ! sptr->ok) {
         if (sptr != NULL)
            delete sptr ;
         errtype = 0 ;
         destroy_progress_window () ;
         return -1 ;
         }
      }
   else
      sptr = NULL ;

/*
   Allocate scratch memory
*/

   if (nhid2)       // Must be REAL model if this is true
      n_grad1 = nhid2 ;
   else if (domain == DOMAIN_COMPLEX_INPUT)
      n_grad1 = nhid1  ?  n_outputs * 2 + nhid1 * 2  :  n_outputs * 2 ;
   else if (domain == DOMAIN_COMPLEX_HIDDEN)
      n_grad1 = n_outputs * 4  +  nhid1 * 4 ;
   else if (domain == DOMAIN_COMPLEX)
      n_grad1 = nhid1  ?  n_outputs * 6  +  nhid1 * 4  :  n_outputs * 4 ;
   else
      n_grad1 = 1 ; // Not used, but allocating something simplifies bookkeeping

   MEMTEXT ( "REGRS_DD: x, 4 works, 2 grads" ) ;
   x = (double *) MALLOC ( ntot * sizeof(double) ) ;
   work1 = (double *) MALLOC ( ntot * sizeof(double) ) ;
   work2 = (double *) MALLOC ( ntot * sizeof(double) ) ;
   if (lptr->method == METHOD_REGRS_CJ) {
      grad2 = (double *) MALLOC ( nout_n * sizeof(double)) ;
      n3 = n4 = ntot ;
      }
   else if (lptr->method == METHOD_REGRS_LM) {
      grad2 = (double *) MALLOC ( ntot * sizeof(double)) ;
      n3 = ntot * ntot ;
      n4 = 1 ; // Not used, but allocating something simplifies bookkeeping
      }
   work3 = (double *) MALLOC ( n3 * sizeof(double) ) ;
   work4 = (double *) MALLOC ( n4 * sizeof(double) ) ;
   grad1 = (double *) MALLOC ( n_grad1 * sizeof(double) ) ;

   if ((x == NULL) || (work1 == NULL) || (work2 == NULL)
    || (work3 == NULL) || (work4 == NULL) || (grad1 == NULL) || (grad2 ==NULL)){
      if (x != NULL)
         FREE ( x ) ;
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
      if (sptr)
         delete sptr ;
      errtype = 0 ;
      destroy_progress_window () ;
      return -1 ;
      }

   local_netptr = this ;
   local_tptr = tptr ;

   memcpy ( x , all_weights , ntot * sizeof(double) ) ; // Regressed values

/*
   Do direct descent optimization, finding local minimum.
*/

   final_accuracy = pow ( 10.0 , -lptr->acc - lptr->refine ) ;

   if (lptr->method == METHOD_REGRS_CJ)
      fval = conjgrad ( 32767 , lptr->quit_err , final_accuracy ,
                        dcrit , ntot , x , neterr , work1 , work2 , work3 ,
                        work4 , lptr->progress ) ;
   else if (lptr->method==METHOD_REGRS_LM)
      fval = lev_marq ( 0 , lptr->quit_err , final_accuracy , lcrit ,
                        ntot , x , sptr , work1 , work2 , work3 ,
                        lptr->progress ) ;

   user_quit = (fval < 0.0) ;
   neterr = fabs ( fval ) ;    // fval<0 if user pressed ESCape

   memcpy ( all_weights , x , ntot * sizeof(double) ) ;

   MEMTEXT ( "REGRS_DD: works (7)" ) ;
   FREE ( x ) ;
   FREE ( work1 ) ;
   FREE ( work2 ) ;
   FREE ( work3 ) ;
   FREE ( work4 ) ;
   FREE ( grad2 ) ;
   if (grad1 != NULL) {
      MEMTEXT ( "REGRS_DD: grad1" ) ;
      FREE ( grad1 ) ;
      }
   if (sptr) {
      MEMTEXT ( "REGRS_DD: delete sptr" ) ;
      delete sptr ;
      }

FINISH:
   sprintf ( msg , "Final error = %.6lf", neterr ) ;
   if (lptr->progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;

   destroy_progress_window () ;

   if (user_quit)
      return 1 ;
   return 0 ;
}
