/******************************************************************************/
/*                                                                            */
/*  ARMA - All routines related to ARMA models                                */
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

#define DEBUG 0

/*
--------------------------------------------------------------------------------

   Constructor

   There is some useful redundancy here.  The number of inputs, MA, and
   outputs can (and will) be computed from inouts.  But it is easy for the
   caller to pass them to the constructor, as they will in all likelihood
   have been computed already.  This lets us do memory allocation immediately.

   We preserve inouts primarily for the user's convenience.  When an ARMA is
   restored from disk, it is lovely to have the audit log file include the
   structure for the sake of a user who failed to keep adequate records.

   In case of malloc failure, we set 'ok' to zero so the user knows about it.

--------------------------------------------------------------------------------
*/

ARMA::ARMA (
   char *armaname ,
   int n_in ,
   int n_ma ,
   int n_out ,
   int n_inouts ,
   InputOutput **inouts ,
   int arma_fixed
   )
{
   int i, j ;

   strcpy ( name , armaname ) ;
   nio = n_inouts ;
   fixed = arma_fixed ;
   ok = 0 ;
   trained = 0 ;
   nw = n_in + n_ma ;
   nvars = fixed ? nw : (nw+1) ;  // Must also estimate constant offset?

   MEMTEXT ( "ARMA constructor (5)" ) ;
   is_input = (int *) MALLOC ( nw * sizeof(int) ) ;
   lag = (int *) MALLOC ( nw * sizeof(int) ) ;
   var = (double *) MALLOC ( n_out * sizeof(double) ) ;
   wts = (double *) MALLOC ( n_out * nvars * sizeof(double) ) ;
   ins_outs = (InputOutput *) MALLOC ( n_inouts * sizeof(InputOutput) ) ;
   if ((is_input == NULL) || (lag == NULL) || (var == NULL)
    || (wts == NULL)  || (ins_outs == NULL)) {
      if (is_input != NULL)
         FREE ( is_input ) ;
      if (lag != NULL)
         FREE ( lag ) ;
      if (var != NULL)
         FREE ( var ) ;
      if (wts != NULL)
         FREE ( wts ) ;
      if (ins_outs != NULL)
         FREE ( ins_outs ) ;
      is_input = lag = NULL ;
      var = wts = NULL ;
      ins_outs = NULL ;
      return ;
      }

   nin = nma = nout = nw = maxlag = 0 ;

	for (i=0 ; i<n_inouts ; i++) {  // Check all ins and outs
      ins_outs[i] = *(inouts[i]) ; // Shallow copy is OK: nothing is deep
	   if (inouts[i]->is_input) {   // If this is an input
         if (inouts[i]->maxlag > maxlag)
            maxlag = inouts[i]->maxlag ;
		   nin += inouts[i]->maxlag - inouts[i]->minlag+1;
         for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++) {
            is_input[nw] = 1 ;
            lag[nw] = j ;
            ++nw ;
            }
         }
		else {                         // This is an output
         for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++) {
            if (! j)       // Only lags are MA terms
               continue ;
            ++nma ;
            is_input[nw] = 0 ;
            lag[nw] = j ;
            ++nw ;
            }
         for (j=0 ; j<i ; j++) {     // Have we seen this output before?
            if ((! inouts[j]->is_input)  &&
                (inouts[i]->which == inouts[j]->which))
               break ;
            }
         if (i == j)       // If this is the first appearance of this output
            ++nout ;  // Count it
         }
      }

   ok = 1 ;
}

/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

ARMA::~ARMA ()
{
   if (! ok)
      return ;

   MEMTEXT ( "ARMA destructor (5)" ) ;
   if (is_input)
      FREE ( is_input ) ;
   if (lag)
      FREE ( lag ) ;
   if (var)
      FREE ( var ) ;
   if (wts)
      FREE ( wts ) ;
   if (ins_outs)
      FREE ( ins_outs ) ;
}

/*
--------------------------------------------------------------------------------

   learn

   Normally this returns 0.  Errors are signaled as follows:
     -1 : Insufficient memory
     -2 : Too few cases
      1 : User pressed ESCape

--------------------------------------------------------------------------------
*/

static double arma_crit ( double *weights ) ;
static int local_nio ;
static InputOutput **local_inouts ;
static Signal **local_signals ;
static int local_ncases ;
static double *local_outvars ;
static int local_nvtot ;
static ARMA *local_arma ;

int ARMA::learn ( int nio , InputOutput **inouts , Signal **signals ,
                  LearnParams *lptr )
{
   int i, j, k, icase, iout, ncases, user_quit ;
   double *outvars, *aptr, *optr, *work, *wptr, err, accuracy ;
   double *work1, *work2 ;
   char msg[400] ;
   Signal *sigptr ;
   SingularValueDecomp *svdptr ;

   user_quit = 0 ;

/*
   Compute the number of cases: Length of the shortest signal minus
   the longest input lag.
*/

   ncases = MAXPOSNUM ;
	for (i=0 ; i<nio ; i++) {  // Check all ins and outs
      sigptr = signals[inouts[i]->which] ;
      if (sigptr->n < ncases)
         ncases = sigptr->n ;
      }

   ncases -= maxlag ;
   if (ncases < (nvars * nout))
      return -2 ;

/*
   Allocate storage for each output variable.

   If there are any MA terms, allocate outvars twice what is needed.
   Get_shocks will use the far half of the vector to save the shocks
   for computing predictions based on the MA terms.
   Technically, we only need to allocate for the MA terms that are present.
   But memory is cheap and time is expensive.  The logic for decoding
   addresses is too slow to justify a little memory savings.
*/

   MEMTEXT ( "ARMA learn: Output variables" ) ;
   if (nma)
      outvars = (double *) MALLOC ( 2 * nout * ncases * sizeof(double) ) ;
   else 
      outvars = (double *) MALLOC ( nout * ncases * sizeof(double) ) ;
   if (outvars == NULL)
      return -1 ;

   make_progress_window ( "ARMA learning" ) ;
   write_progress ( "Initializing..." ) ;

/*
   There are three possibilities.  If there are inputs and the ARMA is not
   already (partially) trained, use regression to estimate the input weights
   and set any MA weights to zero.  Also do this if there are no MA terms,
   as in that case learning is not iterative.
   Else if it is pure MA and not trained, set all MA weights to zero and
   set the constant offset to the mean.
   Finally, regardless of status, we need the outputs in outvars.
*/

   if ((! nma)  ||  (nin  &&  ! trained)) {  // Estimate just input weights

/*
   Allocate the SingularValueDecomposition object for equation solution
   for the input terms.  Also allocate short work vector for solutions.
*/

      MEMTEXT ( "ARMA learn: new SingularValueDecomp, work" ) ;
      svdptr = new SingularValueDecomp ( ncases , fixed ? nin : (nin+1) , 0 ) ;
      work = (double *) MALLOC ( (fixed ? nin : (nin+1)) * sizeof(double) ) ;
      if ((work == NULL)  ||  (svdptr == NULL)  || ! svdptr->ok) {
         if (work != NULL)
            FREE ( work ) ;
         if (svdptr != NULL)
            delete svdptr ;
         FREE ( outvars ) ;
         destroy_progress_window () ;
         return -1 ;
         }

/*
   Fill in the A matrix in svdptr and the output vectors.
   Start at the case offset by the longest lag.
   The columns of A correspond to the weights, with the last column
   being the constant (hence set to 1.0).  For each input, the value
   is the lagged signal.  We skip MA terms for now.
*/

      aptr = svdptr->a ;

      for (icase=maxlag ; icase<ncases+maxlag ; icase++) {// Cases are rows in A
         optr = outvars + icase-maxlag ;// But cases across columns in outvars
         for (i=0 ; i<nio ; i++) {      // Check all ins and outs
            sigptr = signals[inouts[i]->which] ;
   	      if (inouts[i]->is_input) {  // If this is an input
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++)
                  *aptr++ = sigptr->sig[icase-j] ;
               }
            else {      // This is an output
               for (j=0 ; j<i ; j++) {  // Have we seen this output before?
                  if ((! inouts[j]->is_input)  &&
                      (inouts[i]->which == inouts[j]->which))
                     break ;
                  }
               if (i == j) {  // If this is the first appearance of this output
                  *optr = sigptr->sig[icase] ;  // Get this value
                  optr += ncases ; // Cases are strung out as vectors here
                  }
               } // This is an output
            } // For all inputs_outputs
         if (! fixed)
            *aptr++ = 1.0 ;  // Last column is for the constant offset
         } // For all cases

/*
   Decompose the matrix and solve for all weights.
   For each output, copy the output signal to the RHS (b), then put the
   solution in work.  Finally, move the computed weights from work to wts,
   simultaneously filling in any MA weights with zero.
*/

#if DEBUG > 1
      for (icase=0 ; icase<ncases ; icase++) {
         printf ( "\nInput case %d: ", icase ) ;
         for (j=0 ; j<(fixed ? nin : (nin+1)) ; j++)
            printf ( " %12.6lf", svdptr->a[icase*(fixed ? nin : (nin+1))+j] ) ;
         printf ( " : " ) ;
         for (j=0 ; j<nout ; j++)
            printf ( " %12.6lf", outvars[j*ncases+icase] ) ;
         }
#endif

      svdptr->svdcmp () ;
      for (iout=0 ; iout<nout ; iout++) {
         memcpy ( svdptr->b , outvars+iout*ncases , ncases * sizeof(double) ) ;
         svdptr->backsub ( 1.e-8 , work ) ;

#if DEBUG
         printf ( "\nOUTPUT %d:", iout ) ;
         for (j=0 ; j<(fixed ? nin : (nin+1)) ; j++)
            printf ( " %lf", work[j] ) ;
#endif

         wptr = work ;                  // The input weights are here
         optr = wts + iout * nvars ;    // Move them to here
         for (i=0 ; i<nio ; i++) {      // Check all ins and outs
   	      if (inouts[i]->is_input) {  // If this is an input
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++)
                  *optr++ = *wptr++ ;
               }
            else {
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++) {
                  if (! j)       // Only lags are MA terms
                     continue ;
                  *optr++ = 0.0 ;
                  }
               } // MA term
            } // For all inouts
         if (! fixed)
            *optr = *wptr ;         // This is the constant term
         } // For all outputs

      MEMTEXT ( "ARMA learn: delete SingularValueDecomp, work" ) ;
      delete svdptr ;
      FREE ( work ) ;
      } // If nin, solve for input weights

/*
   If there are no inputs, set all the MA weights to zero and set
   the constant offset to the mean.
*/

   else if (! trained) {

      for (iout=0 ; iout<nout ; iout++)    // Borrow variance vector
         var[iout] = 0.0 ;                 // For summing means

      for (icase=maxlag ; icase<ncases+maxlag ; icase++) {
         optr = outvars + icase-maxlag ;   // Offset outputs
         k = 0 ;                           // Count unique outputs
         for (i=0 ; i<nio ; i++) {         // These guaranteed all outputs
            for (j=0 ; j<i ; j++) {        // Have we seen this output before?
               if ((! inouts[j]->is_input)  &&
                   (inouts[i]->which == inouts[j]->which))
                  break ;
               }
            if (i == j) {  // If this is the first appearance of this output
               sigptr = signals[inouts[i]->which] ;
               *optr = sigptr->sig[icase] ; // Get this value
               var[k++] += *optr ;  // Sum means
               optr += ncases ;     // Cases are strung out as vectors here
               }
            } // For all inputs_outputs
         } // For all cases

      wptr = wts ;  // Set all MA weights to zero, constants to mean
      for (iout=0 ; iout<nout ; iout++) {  // For each predicted outut
         for (i=0 ; i<nma ; i++)           // Each MA weight
            *wptr++ = 0.0 ;                // is zero
         if (! fixed)
            *wptr++ = var[iout] / ncases ; // And constant is the mean
         }

      } // Else pure MA and not trained

/*
   Else already trained.  Just put outputs in outvars.
*/

   else
      get_outputs ( nio , inouts , signals , ncases , 0 , 0 , 0 , outvars ) ;

#if DEBUG
   for (iout=0 ; iout<nout ; iout++) {
      printf ( "\nInit OUTPUT %d:", iout ) ;
      for (j=0 ; j<nvars ; j++)
         printf ( " %lf", wts[iout*nvars+j] ) ;
      }
#endif

/*
   Initialization is practically done.  The weights are in wts.
   All input weights have been found by regression.  All MA weights are zero.
   The constant offset has been found.
   The true outputs are in outvars.  Note that outvars[0] is for time 'maxlag'.
   This offset is because earlier predictions cannot be made.
   Complete initialization by computing the errors (shocks).  Put these
   in the far half of outvars.
*/

   (void) get_shocks ( nio , inouts , signals , ncases , outvars ) ;

#if DEBUG
   printf ( "\nError=%lf : ", error ) ;
   for (iout=0 ; iout<nout ; iout++)
      printf ( " %lf", var[iout] ) ;
   getch () ;
#endif

/*
   If there are no MA terms, we are done.
*/

   sprintf ( msg , "Initialization error = %lf", error ) ;
   if (lptr->progress)
      write_progress ( msg ) ;
   else 
      write_non_progress ( msg ) ;

   if (! nma) {
      trained = 1 ;
      goto FINISH ;
      }

/*
   Iteratively minimize the grand mean error.
*/

   MEMTEXT ( "ARMA learn: work1, 2" ) ;
   work1 = (double *) MALLOC ( 3 * nvars * nout * sizeof(double) ) ;
   work2 = (double *) MALLOC ( nvars * nout * nvars * nout * sizeof(double) ) ;
   if ((work1 == NULL)  ||  (work2 == NULL)) {
      destroy_progress_window () ;
      return -1 ;
      }

   local_nio = nio ;
   local_inouts = inouts ;
   local_signals = signals ;
   local_ncases = ncases ;
   local_outvars = outvars ;
   local_nvtot = nvars * nout ;
   local_arma = this ;

   accuracy = pow ( 10.0 , -lptr->acc - lptr->refine ) ;
   err = error ;
   memcpy ( work1 , wts , nvars * nout * sizeof(double) ) ;

   write_progress ( "Refining..." ) ;
   error = powell ( 0 , lptr->quit_err , accuracy ,
                    arma_crit , nvars * nout , work1 , error ,
                    work1 + nvars * nout , work1 + 2 * nvars * nout ,
                    work2 , lptr->progress ) ;

   memcpy ( wts , work1 , nvars * nout * sizeof(double) ) ;

   if ((error >= 0.0)  ||  (-error < err))  // This is pretty subjective.
      trained = 1 ;       // Consider it trained if not interrupted or improved

   (void) get_shocks ( nio , inouts , signals , ncases , outvars ) ;

#if DEBUG
   printf ( "\nFinal Error=%lf : ", error ) ;
   for (iout=0 ; iout<nout ; iout++)
      printf ( " %lf", var[iout] ) ;
   for (iout=0 ; iout<nout ; iout++) {
      printf ( "\nOUTPUT %d:", iout ) ;
      for (j=0 ; j<nvars ; j++)
         printf ( " %lf", wts[iout*nvars+j] ) ;
      }
   getch () ;
#endif

   MEMTEXT ( "ARMA learn: work1, 2" ) ;
   FREE ( work1 ) ;
   FREE ( work2 ) ;

   if (error < 0.0) {
      error = -error ;
      user_quit = 1 ;
      }

FINISH:
   MEMTEXT ( "ARMA learn: outvars" ) ;
   FREE ( outvars ) ;

   destroy_progress_window () ;
   return user_quit ;
}

/*
--------------------------------------------------------------------------------

   get_shocks - Compute the shocks (and cumulate their variance)

   If there are MA terms, outvars must be allocated twice as long.
   The second half will be used for scratch storage of shocks.

--------------------------------------------------------------------------------
*/

double ARMA::get_shocks ( int nio , InputOutput **inouts , Signal **signals ,
                          int ncases , double *outvars )
{
   int i, j, k, icase, iout ;
   double err, *optr, *wptr, *dptr, *shocks ;

   for (iout=0 ; iout<nout ; iout++)      // Will sum error variances here
      var[iout] = 0.0 ;

   for (icase=0 ; icase<ncases ; icase++) {
      optr = outvars + icase ;            // Offset by maxlag later
      shocks = optr + nout * ncases ;     // Shocks go here (if MA terms)
      for (iout=0 ; iout<nout ; iout++) { // For each predicted output
         wptr = wts + iout * nvars ;      // Weights for this output here
         err = 0.0 ;                      // Will cumulate predicted here
         for (i=0 ; i<nio ; i++) {        // Check all ins and outs
	         if (inouts[i]->is_input) {    // If this is an input
               dptr = signals[inouts[i]->which]->sig + icase + maxlag ;
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++)
                  err += *(dptr-j) * *wptr++ ; // Sum prediction
               }
            else {      // This is an output
               k = inouts[i]->ordinal ;   // Output vector for this MA term
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++) {
                  if (! j)              // Only lags are MA terms
                     continue ;
                  if (j <= icase)       // If this shock has been computed
                     err += shocks[k*ncases-j] * *wptr++ ; // Sum prediction
                  else                  // Shocks before the first prediction
                     ++wptr ;           // Are assumed zero
                  }
               } // This is an output
            } // For all inputs_outputs
         if (! fixed)
            err += *wptr ;              // Last weight is the constant offset
         err = optr[iout*ncases] - err ;// Shock is actual minus predicted
         if (err > 1.e30)               // Prevent runaway
            err = 1.e30 ;
         if (err < -1.e30)
            err = -1.e30 ;
         if (nma)
            shocks[iout*ncases] = err ; // Save it for subsequent predictions
         var[iout] += err * err ;       // Cumulate error
         } // For all predicted output signals
      } // For all cases

   error = 0.0 ;                        // This will be grand mean error
   for (iout=0 ; iout<nout ; iout++) {  // For each predicted output
      var[iout] /= ncases ;             // Compute its mean error
      error += var[iout] ;              // And grand error
      }
   error /= nout ;
   return error ;
}

static double arma_crit ( double *x )
{
   memcpy ( local_arma->wts , x , local_nvtot * sizeof(double) ) ;
   return local_arma->get_shocks ( local_nio , local_inouts , local_signals ,
                                   local_ncases , local_outvars ) ;
}

/*
--------------------------------------------------------------------------------

   get_outputs - Fill outvars with the signal values.  Use 1.e40 for
                 cases beyond the end of the signal if 'flag_end' is true.
                 Otherwise do not do anything beyond the end.
                 Start outputs at first point if 'from_start' is nonzero.
                 Else start it offset by maxlag.

--------------------------------------------------------------------------------
*/
void ARMA::get_outputs ( int nio , InputOutput **inouts , Signal **signals ,
                         int ncases , int flag_end , int from_start ,
                         int get_shocks , double *outvars )
{
   int i, j, ishock, icase ;
   double *optr, *shocks ;
   Signal *sigptr ;

   for (icase=0 ; icase<ncases ; icase++) {
      optr = outvars + icase ;
      shocks = optr + nout * ncases ;  // Shocks go here (if get_shocks true)
      for (i=0 ; i<nio ; i++) {        // Check all ins and outs
         if (inouts[i]->is_input)      // If this is an input
            continue ;                 // Ignore it
         for (j=0 ; j<i ; j++) {       // Have we seen this output before?
            if ((! inouts[j]->is_input)  &&
                (inouts[i]->which == inouts[j]->which))
               break ;
            }
         if (i == j) {  // If this is the first appearance of this output
            j = from_start  ?  icase  :  icase+maxlag ;
            sigptr = signals[inouts[i]->which] ;
            if (j < sigptr->n)
               *optr = sigptr->sig[j] ; // Get this value
            else if (flag_end)
               *optr = 1.e40 ;   // Flag that we are beyond the end
            optr += ncases ;     // Cases are strung out as vectors here
            if (! get_shocks)
               continue ;
            *shocks = 1.e40 ;    // Default value forces computation in predict
            ishock = inouts[i]->shock ;
            if (ishock >= 0) {
               sigptr = signals[ishock] ;
               if (j < sigptr->n)
                  *shocks = sigptr->sig[j] ; // Get this value
               }
            shocks += ncases ;   // Cases are strung out as vectors here
            }
         } // For all inputs_outputs
      } // For all cases
}

/*
--------------------------------------------------------------------------------

   predict - Predict beyond the end of the series

   Be sure to call get_outputs first to initialize outvars.
   Also, initialize all outputs beyond the known signal to >1.e35 as a flag
   that this is virgin territory.

   If the shocks are known, initialize them.  Otherwise set them >1.e35 as
   a flag to compute them here.

   Note that ncases is the number of predictions.  The total length of outvars
   is ntot = ncases + maxlag, as the first maxlag cases remain untouched.

   For input terms, there are three possibilities:
     1) The term is still within the input extent --- Use it.
     2) The term is beyond the end, and we are predicting this input ---
        get it as a previous recursive prediction
     3) The term is beyond the end and we are not predicting it ---
        Duplicate the final point in the series.  This is rare and displeasing.

   If there are MA terms, outvars must be allocated twice as long.
   The second half will be used for scratch storage of shocks.

--------------------------------------------------------------------------------
*/

double ARMA::predict ( int nio , InputOutput **inouts , Signal **signals ,
                       int ncases , int shocks_known , double *outvars )
{
   int i, j, k, icase, iout, ntot ;
   double pred, *optr, *wptr, *dptr, *shocks, err ;
   Signal *sigptr ;

   ntot = ncases + maxlag ;  // First maxlag outputs remain at original values

   for (iout=0 ; iout<nout ; iout++)      // Will sum error variances here
      var[iout] = 0.0 ;

   for (icase=0 ; icase<ncases ; icase++) {
      optr = outvars + icase + maxlag ;   // Keep first maxlag at original vals
      shocks = optr + nout * ntot ;       // Shocks go here (if MA terms)
      for (iout=0 ; iout<nout ; iout++) { // For each predicted output
         wptr = wts + iout * nvars ;      // Weights for this output here
         pred = 0.0 ;                     // Will cumulate predicted here
         for (i=0 ; i<nio ; i++) {        // Check all ins and outs
	         if (inouts[i]->is_input) {    // If this is an input
               sigptr = signals[inouts[i]->which] ;
               dptr = sigptr->sig + icase + maxlag ;
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++) {
                  if (icase+maxlag-j < sigptr->n)  // Common: within signal
                     pred += *(dptr-j) * *wptr++ ; // Sum prediction
                  else if ((k = inouts[i]->is_other) >= 0) {  // Recursive
                     if (j <= icase)       // Normally true, but insurance
                        pred += optr[k*ncases-j] * *wptr++ ; // Sum prediction
                     }
                  else  // Rare situation: beyond end of unpredicterd signal
                     pred += *(sigptr->sig + sigptr->n - 1) * *wptr++ ; // Dup
                  }
               }
            else {      // This is an output
               k = inouts[i]->ordinal ; // Output vector for this MA term
               for (j=inouts[i]->minlag ; j<=inouts[i]->maxlag ; j++) {
                  if (! j)              // Only lags are MA terms
                     continue ;
                  if (j <= icase)       // If this shock is available
                     pred += shocks[k*ntot-j] * *wptr++ ; // Sum prediction
                  else                  // Shocks before the first prediction
                     ++wptr ;           // Are assumed zero
                  }
               } // This is an output
            } // For all inputs_outputs
         if (! fixed)
            pred += *wptr ;             // Last weight is the constant offset
         if (pred > 1.e30)              // Prevent runaway
            pred = 1.e30 ;
         if (pred < -1.e30)
            pred = -1.e30 ;
         if (optr[iout*ntot] < 1.e35) {      // If this is still known signal
            err = optr[iout*ntot] - pred ;   // Then this makes sense
            var[iout] += err * err ;         // Cumulate error
            }
         else {                              // We are now beyond the end
            err = 0.0 ;                      // Assume no errors after known
            if (var[iout] >= 0.0)            // True the first time end passed
               var[iout] = -var[iout] / icase ; // Find the mean for this out
            }                                // And flag to not divide again
         if (nma  ||  shocks_known) {        // If there is a shock vector
            if (shocks[iout*ntot] < 1.e35)   // If this shock is known
               pred += shocks[iout*ntot] ;   // Use it to improve prediction
            else if (nma)                    // Unknown.  If there are MA terms
               shocks[iout*ntot] = err ;     // Must compute the shocks
            }
         optr[iout*ntot] = pred ;            // Record the prediction
         } // For all predicted output signals
      } // For all cases

   error = 0.0 ;                        // This will be grand mean error
   for (iout=0 ; iout<nout ; iout++) {  // For each predicted output
      if (var[iout] < 0.0)              // If there were predictions past end
         var[iout] = -var[iout] ;       // Mean was already computed
      else 
         var[iout] /= ncases ;          // Compute its mean error
      error += var[iout] ;              // And grand error
      }
   error /= nout ;
   return error ;
}
