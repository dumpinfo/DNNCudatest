/******************************************************************************/
/*                                                                            */
/*  TRAIN - All routines related to cumulating a training set                 */
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
#include "const.h"     // System, limitation constants, typedefs, structs
#include "classes.h"   // Includes all class headers
#include "funcdefs.h"  // Function prototypes


/*
--------------------------------------------------------------------------------

   Constructor

--------------------------------------------------------------------------------
*/

TrainingSet::TrainingSet (
   int out_model ,
   int nin ,
   int nout ,
   int n_inputs_outputs ,
   InputOutput **inputs_outputs
   )
{
   int i, j, lag ;
   InputOutput *ioptr ;

   output_mode = out_model ;

   n_inputs = nin ;
   n_outputs = nout ;

/*
   Compute the size of each training sample.
   CLASSIFY output uses a double at the end of each input sample to indicate
   the class membership (0=reject).
   If this is CLASSIFY mode, we also allocate an array that keeps track of
   how many samples are in each class.  This is typically very small, so the
   malloc should not fail.  Nevertheless, we check for that failure when we
   call 'train' to build the training set.  We do the same for the array
   of prior probabilities.
   Note that we allocate at least 1.  This kludge is in case there are
   only REJECT cases so far, so that n_outputs=0.
   We also allocate arrays for the lag (of inputs) and the lead (of outputs
   if not CLASSIFY mode).  When a network is trained, this information is
   copied to it.  When it is tested, a comparison with the test set is done
   to catch user errors.  Also, the lead is used in prediction.
*/

   nper = NULL ;         // Used in CLASSIFY mode to count cases in each class
   priors = NULL ;       // Used in CLASSIFY mode for prior probability weights
   lags = leads = NULL ; // Passed to trained network

   ntrain = 0 ;
   data = NULL ;

   MEMTEXT ( "TRAIN constructor: lags" ) ;
   lags = (unsigned *) MALLOC ( n_inputs * sizeof(unsigned) ) ;
   if (lags == NULL)
      return ;

   j = 0 ;
   for (i=0 ; i<n_inputs_outputs ; i++) {
      ioptr = inputs_outputs[i] ;
      if (ioptr->is_input) {
         for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++)
            lags[j++] = lag ;
         }
      }

   if (output_mode == OUTMOD_CLASSIFICATION) {
      size = n_inputs + 1 ;
      MEMTEXT ( "TRAIN constructor: nper, priors" ) ;
      nper = (unsigned int *) MALLOC ( (n_outputs ? n_outputs : 1) * sizeof(unsigned) ) ;
      if (nper != NULL) {
         memset ( nper , 0 , n_outputs * sizeof(unsigned) ) ;
         priors = (double *) MALLOC ( (n_outputs ? n_outputs : 1) * sizeof(double) ) ;
         if (priors == NULL) {
            FREE ( nper ) ;
            nper = NULL ;
            }
         }
      else
         priors = NULL ;
      }
   else if (output_mode == OUTMOD_MAPPING) {
      size = n_inputs + n_outputs ;
      MEMTEXT ( "TRAIN constructor: leads" ) ;
      leads = (unsigned *) MALLOC ( n_outputs * sizeof(unsigned) ) ;
      if (leads == NULL)
         return ;
      j = 0 ;
      for (i=0 ; i<n_inputs_outputs ; i++) {
         ioptr = inputs_outputs[i] ;
         if (! ioptr->is_input) {
            for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++)
               leads[j++] = lag ;
            }
         }
      }
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

TrainingSet::~TrainingSet ()
{
   if (nper != NULL) {
      MEMTEXT ( "TRAIN: nper" ) ;
      FREE ( nper ) ;
      }
   if (priors != NULL) {
      MEMTEXT ( "TRAIN: priors" ) ;
      FREE ( priors ) ;
      }
   if (data != NULL) {
      MEMTEXT ( "TRAIN: data" ) ;
      FREE ( data ) ;
      }
   if (lags != NULL) {
      MEMTEXT ( "TRAIN: lags" ) ;
      FREE ( lags ) ;
      }
   if (leads != NULL) {
      MEMTEXT ( "TRAIN: leads" ) ;
      FREE ( leads ) ;
      }
}

/*
--------------------------------------------------------------------------------

   Assignment operator

--------------------------------------------------------------------------------
*/

void TrainingSet::operator= ( const TrainingSet& tset )
{
   if (this == &tset)
      return ;

   if (nper != NULL) {
      MEMTEXT ( "TRAIN: = nper" ) ;
      FREE ( nper ) ;
      nper = NULL ;
      }
   if (priors != NULL) {
      MEMTEXT ( "TRAIN: = priors" ) ;
      FREE ( priors ) ;
      priors = NULL ;
      }
   if (data != NULL) {
      MEMTEXT ( "TRAIN: = data" ) ;
      FREE ( data ) ;
      data = NULL ;
      }
   if (lags != NULL) {
      MEMTEXT ( "TRAIN: = lags" ) ;
      FREE ( lags ) ;
      lags = NULL ;
      }
   if (leads != NULL) {
      MEMTEXT ( "TRAIN: = leads" ) ;
      FREE ( leads ) ;
      leads = NULL ;
      }

   output_mode = tset.output_mode ;
   n_inputs = tset.n_inputs ;
   n_outputs = tset.n_outputs ;
   size = tset.size ;

   ntrain = 0 ;


   MEMTEXT ( "TRAIN: = lags" ) ;
   lags = (unsigned *) MALLOC ( tset.n_inputs * sizeof(unsigned) ) ;
   if (lags == NULL)
      return ;
   memcpy ( lags , tset.lags , tset.n_inputs * sizeof(unsigned) ) ;

   if (tset.ntrain) {
      MEMTEXT ( "TRAIN: = data" ) ;
      data = (double *) MALLOC ( tset.ntrain * size * sizeof(double) ) ;
      if (data == NULL) {
         FREE ( lags ) ;
         return ;
         }
      }

   if (output_mode == OUTMOD_CLASSIFICATION) {
      MEMTEXT ( "TRAIN: = nper, priors" ) ;
      nper = (unsigned int *) MALLOC ( tset.n_outputs * sizeof(unsigned) ) ;
      priors = (double *) MALLOC ( tset.n_outputs * sizeof(double) ) ;
      if ((nper == NULL)  ||  (priors == NULL)) {
         if (nper != NULL)
            FREE ( nper ) ;
         if (priors != NULL)
            FREE ( priors ) ;
         if (data != NULL)
            FREE ( data ) ;
         FREE ( lags ) ;
         return ;
         }
      memcpy ( nper , tset.nper , n_outputs * sizeof(unsigned) ) ;
      memcpy ( priors , tset.priors , n_outputs * sizeof(double) ) ;
      }
   else if (output_mode == OUTMOD_MAPPING) {
      MEMTEXT ( "TRAIN: = leads" ) ;
      leads = (unsigned *) MALLOC ( tset.n_outputs * sizeof(unsigned) ) ;
      if (leads == NULL) {
         if (data != NULL)
            FREE ( data ) ;
         FREE ( lags ) ;
         return ;
         }
      memcpy ( leads , tset.leads , tset.n_outputs * sizeof(unsigned) ) ;
      }

   ntrain = tset.ntrain ;
   if (ntrain)
      memcpy ( data , tset.data , ntrain * size * sizeof(double) ) ;
}


/*
--------------------------------------------------------------------------------

   train - Add members to a training set

--------------------------------------------------------------------------------
*/

int TrainingSet::train (
   NetParams *net_params ,
   MiscParams *misc_params ,
   int n_inputs_outputs ,
   InputOutput **inputs_outputs ,
   Signal **signals
   )
{
   int i, j, casenum, ncases, start, stop, shortest, bytes, lag ;
   unsigned int *iptr ;
   double *dptr ;
   Signal *sigptr ;
   InputOutput *ioptr ;

/*
   If this is CLASSICATION mode, the constructor allocated an array for holding
   the number of cases in each class.  It did not check for the unlikely event
   of failure of that malloc.  Verify that it went ok.  Ditto for prior, lag.
   And in MAPPING mode, check lead.
*/

   if (lags == NULL)
      return 1 ;

   if (output_mode == OUTMOD_CLASSIFICATION) {
      if ((nper == NULL)  ||  (priors == NULL))
         return 1 ;
      }

   if (output_mode == OUTMOD_MAPPING) {
      if (leads == NULL)
         return 1 ;
      }

/*
   Verify that the essential characteristics of this appended training set
   are compatible with what already exists.
*/

   if ((net_params->out_model != output_mode)
    || (net_params->n_inputs != n_inputs)
    || ((output_mode != OUTMOD_CLASSIFICATION) && (net_params->n_outputs != n_outputs)))
      return 2 ;

   j = 0 ;
   for (i=0 ; i<n_inputs_outputs ; i++) {
      ioptr = inputs_outputs[i] ;
      if (ioptr->is_input) {
         for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++)
            if (lags[j++] != lag)
               return 2 ;
         }
      }

   if (output_mode == OUTMOD_MAPPING) {
      j = 0 ;
      for (i=0 ; i<n_inputs_outputs ; i++) {
         ioptr = inputs_outputs[i] ;
         if (! ioptr->is_input) {
            for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++)
               if (leads[j++] != lag)
                  return 2 ;
            }
         }
      }

/*
   If the number of classes has increased, a common event, allocate more
   space for nper and priors.
*/

   if ((output_mode == OUTMOD_CLASSIFICATION)  &&  (net_params->n_outputs > n_outputs)) {
      MEMTEXT ( "TRAIN appending class: nper, priors" ) ;
      iptr = (unsigned int *) REALLOC (
                        nper , net_params->n_outputs * sizeof(unsigned int) ) ;
      if (iptr != NULL) {
         nper = iptr ;
         while (n_outputs < net_params->n_outputs)
            nper[n_outputs++] = 0 ;
         dptr = (double *) REALLOC ( priors , n_outputs * sizeof(double) ) ;
         if (dptr == NULL) {
            FREE ( nper ) ;
            nper = NULL ;
            return 1 ;
            }
         priors = dptr ;
         }
      }

/*
   Compute the starting and ending positions, and the number of cases.
*/

   start = stop = 0 ;
   shortest = MAXPOSNUM ;
   for (i=0 ; i<n_inputs_outputs ; i++) {
      ioptr = inputs_outputs[i] ;
      sigptr = signals[ioptr->which] ;
      if (sigptr->n < shortest)
         shortest = sigptr->n ;
      if (ioptr->is_input) {
         if (ioptr->maxlag > start)
            start = ioptr->maxlag ;
         }
      else {
         if (ioptr->maxlag > stop)
            stop = ioptr->maxlag ;
         }
      }

   start += misc_params->exclude ;    // Exclude this many bad cases
   ncases = shortest - start - stop ; // Total number to cumulate

   if (ncases > misc_params->include)
      ncases = misc_params->include ;

   if (ncases <= 0)
      return 3 ;

/*
   Allocate for this new data
*/

   bytes = (ntrain + ncases) * size * sizeof(double) ;
   if (data == NULL) {
      MEMTEXT ( "TRAIN: initial" ) ;
      dptr = (double *) MALLOC ( bytes ) ;
      }
   else {
      MEMTEXT ( "TRAIN: append" ) ;
      dptr = (double *) REALLOC ( data , bytes ) ;
      }
   if (dptr == NULL)
      return 1 ;
   data = dptr ;

/*
   Collect the training set
*/

   dptr = data + ntrain * size ;      // New cases will start here
   for (casenum=0 ; casenum<ncases ; casenum++) {

      for (i=0 ; i<n_inputs_outputs ; i++) { // Inputs first
         ioptr = inputs_outputs[i] ;
         if (ioptr->is_input) {
            sigptr = signals[ioptr->which] ;
            for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++)
               *dptr++ = (sigptr->sig)[casenum+start-lag] ;
            }
         }

      if (output_mode == OUTMOD_CLASSIFICATION)   // Outputs follow inputs
         *dptr++ = (double) misc_params->classif_output + 0.1 ;

      else if (output_mode == OUTMOD_MAPPING) {
         for (i=0 ; i<n_inputs_outputs ; i++) {
            ioptr = inputs_outputs[i] ;
            if (! ioptr->is_input) {
               sigptr = signals[ioptr->which] ;
               for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++)
                  *dptr++ = (sigptr->sig)[casenum+start+lag] ;
               }
            }
         }
      }  // Append all new cases

/*
   Final cleanup adds in these new cases
*/

   ntrain += ncases ;

   if (output_mode == OUTMOD_CLASSIFICATION) {
      if (misc_params->classif_output) {    // No count kept for REJECT
         nper[misc_params->classif_output-1] += ncases ;
         priors[misc_params->classif_output-1] = misc_params->classif_prior ;
         }
      }

   return 0 ;
}

