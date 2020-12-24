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
/*  LAYERNET - All principal routines for LayerNet processing                 */
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

static void free_non_null ( void **p ) ;

/*
--------------------------------------------------------------------------------

   Constructor

   The parameter 'executable' determines whether work areas for hidden and
   output neuron activations are also allocated.  These are needed if we
   will ever apply inputs and want to compute outputs.
   In case of malloc failure, we set 'ok' to zero so the user knows about it.
   Also, we always leave unallocated weight pointers set to NULL.  There is no
   hard reason for doing this; calling programs should always know enough not
   to reference them.  However, it is simply good style.  Most compilers are
   much better at producing code that intercepts NULL pointer references than
   just wild pointers.  An ounce of prevention...

--------------------------------------------------------------------------------
*/

LayerNet::LayerNet (
   int net_model ,
   int out_model ,
   int out_linear ,
   int n_inputs ,
   int n_hidden1 ,
   int n_hidden2 ,
   int n_outputs ,
   int executable ,  // Also allocate hidden and output neurons?
   int zero          // Zero all weights?
   )
{
   int n1, n2, n3 ;

   model = net_model ;
   errtype = 0 ;
   outmod = out_model ;
   outlin = out_linear ;
   nin = n_inputs ;
   nhid1 = n_hidden1 ;
   nhid2 = n_hidden2 ;
   nout = n_outputs ;
   exe = executable ;
   neterr = 1.e30 ;

   if (model == NETMOD_REAL) {
      nin_n = nin + 1 ;
      nhid1_n = nhid1 + 1 ;
      nhid2_n = nhid2 + 1 ;
      nout_n = nout ;
      nin_w = nin ;
      nhid1_w = nhid1 ;
      nhid2_w = nhid2 ;
      }
   else if (model == NETMOD_COMPLEX) {
      nin_n = 2 * nin + 2 ;
      nhid1_n = 2 * nhid1 + 2 ;
      nhid2_n = 2 * nhid2 + 2 ;
      nout_n = 2 * nout ;
      nin_w = 2 * nin ;
      nhid1_w = 2 * nhid1 ;
      nhid2_w = 2 * nhid2 ;
      }
   else if (model == NETMOD_COMPLEX_INPUT) {
      nin_n = 2 * nin + 2 ;
      nhid1_n = nhid1 + 1 ;
      nhid2_n = nhid2 + 1 ;
      nout_n = nout ;
      nin_w = 2 * nin ;
      nhid1_w = nhid1 ;
      nhid2_w = nhid2 ;
      }
   else if (model == NETMOD_COMPLEX_HIDDEN) {
      nin_n = 2 * nin + 2 ;
      nhid1_n = 2 * nhid1 + 2 ;
      nhid2_n = 2 * nhid2 + 2 ;
      nout_n = nout ;
      nin_w = 2 * nin ;
      nhid1_w = 2 * nhid1 ;
      nhid2_w = 2 * nhid2 ;
      }

   confusion = NULL ;
   hid1_coefs = hid2_coefs = out_coefs = hid1 = hid2 = out = NULL ;

   ok = 0 ;   // Indicates failure of malloc (What a pessimist!)

   if (exe  &&  (outmod == OUTMOD_CLASSIFY)  &&
       (confusion = (int *) MALLOC ( (nout+1) * sizeof(int))) == NULL)
      return ; // One for each class, plus reject

   if (nhid1 == 0) {                // No hidden layer
      n1 = nout * nin_n ;
      if (((out_coefs = (double *) MALLOC ( n1 * sizeof(double) )) == NULL)
       || (exe && (out = (double *) MALLOC (nout_n * sizeof(double))) == NULL)){
         free_non_null ( (void **) &out_coefs ) ;
         free_non_null ( (void **) &confusion ) ;
         return ;
         }
      if (zero) {
         while (n1--)
            out_coefs[n1] = 0.0 ;
         }
      }

   else if (nhid2 == 0) {           // One hidden layer
      n1 = nhid1 * nin_n ;
      n2 = nout * nhid1_n ;
      if (((hid1_coefs = (double *) MALLOC ( n1 * sizeof(double) )) == NULL)
       || ((out_coefs = (double *) MALLOC ( n2 * sizeof(double) ))==NULL)
       || (exe && (hid1 = (double *) MALLOC ( nhid1_w * sizeof(double) ))==NULL)
       || (exe && (out = (double *) MALLOC (nout_n * sizeof(double))) == NULL)){
         free_non_null ( (void **) &hid1_coefs ) ;
         free_non_null ( (void **) &out_coefs ) ;
         free_non_null ( (void **) &hid1 ) ;
         free_non_null ( (void **) &confusion ) ;
         return ;
         }
      if (zero) {
         while (n1--)
            hid1_coefs[n1] = 0.0 ;
         while (n2--)
            out_coefs[n2] = 0.0 ;
         }
      }

   else {                           // Two hidden layers
      n1 = nhid1 * nin_n ;
      n2 = nhid2 * nhid1_n ;
      n3 = nout * nhid2_n ;
      if (((hid1_coefs = (double *) MALLOC ( n1 * sizeof(double) )) == NULL)
       || ((hid2_coefs = (double *) MALLOC ( n2 * sizeof(double) )) == NULL)
       || ((out_coefs = (double *) MALLOC ( n3 * sizeof(double) ))==NULL)
       || (exe && (hid1 = (double *) MALLOC ( nhid1_w * sizeof(double) ))==NULL)
       || (exe && (hid2 = (double *) MALLOC ( nhid2_w * sizeof(double) ))==NULL)
       || (exe && (out = (double *) MALLOC (nout_n * sizeof(double))) == NULL)){
         free_non_null ( (void **) &hid1_coefs ) ;
         free_non_null ( (void **) &hid2_coefs ) ;
         free_non_null ( (void **) &out_coefs ) ;
         free_non_null ( (void **) &hid1 ) ;
         free_non_null ( (void **) &hid2 ) ;
         free_non_null ( (void **) &confusion ) ;
         return ;
         }
      if (zero) {
         while (n1--)
            hid1_coefs[n1] = 0.0 ;
         while (n2--)
            hid2_coefs[n2] = 0.0 ;
         while (n3--)
            out_coefs[n3] = 0.0 ;
         }
      }

   if (confusion != NULL)
      memset ( confusion , 0 , (nout+1) * sizeof(int) ) ;

   ok = 1 ;            // Indicate to caller that all mallocs succeeded
}

/*
   Local routine to free non-null pointers
*/

static void free_non_null ( void **p )
{
   if (*p != NULL) {
      FREE ( *p ) ;
      *p = NULL ;
      }
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

LayerNet::~LayerNet()
{
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   FREE ( out_coefs ) ;
   if (exe) {
      FREE ( out ) ;
      if (confusion != NULL)
         FREE ( confusion ) ;
      }

   if (nhid1) {
      FREE ( hid1_coefs ) ;
      if (exe)
         FREE ( hid1 ) ;
      if (nhid2) {
         FREE ( hid2_coefs ) ;
         if (exe)
            FREE ( hid2 ) ;
         }
      }
}

/*
--------------------------------------------------------------------------------

   copy_weights - Copy the weights from one network to another
                  Note that this is NOT like a copy or assignment,
                  as it does not copy other parameters.  In fact,
                  it gets sizes from the calling instance!

--------------------------------------------------------------------------------
*/

void LayerNet::copy_weights ( LayerNet *dest , LayerNet *source )
{
   int n ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      memcpy ( dest->out_coefs , source->out_coefs , n * sizeof(double) ) ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      memcpy ( dest->hid1_coefs , source->hid1_coefs , n * sizeof(double) ) ;
      n = nout * nhid1_n ;
      memcpy ( dest->out_coefs , source->out_coefs , n * sizeof(double) ) ;
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      memcpy ( dest->hid1_coefs , source->hid1_coefs , n * sizeof(double) ) ;
      n = nhid2 * nhid1_n ;
      memcpy ( dest->hid2_coefs , source->hid2_coefs , n * sizeof(double) ) ;
      n = nout * nhid2_n ;
      memcpy ( dest->out_coefs , source->out_coefs , n * sizeof(double) ) ;
      }
}


/*
--------------------------------------------------------------------------------

   zero_weights - Zero all weights in a network

--------------------------------------------------------------------------------
*/

void LayerNet::zero_weights ()
{
   int n ;

   neterr = 1.e30 ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      while (n--)
         out_coefs[n] = 0.0 ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      while (n--)
         hid1_coefs[n] = 0.0 ;
      n = nout * nhid1_n ;
      while (n--)
         out_coefs[n] = 0.0 ;
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      while (n--)
         hid1_coefs[n] = 0.0 ;
      n = nhid2 * nhid1_n ;
      while (n--)
         hid2_coefs[n] = 0.0 ;
      n = nout * nhid2 ;
      while (n--)
         out_coefs[n] = 0.0 ;
      }
}


/*
--------------------------------------------------------------------------------

   trial - Compute the output for a given input by evaluating network

--------------------------------------------------------------------------------
*/

void LayerNet::trial ( double *input )
{
   int i ;
   
   if (! exe) {   // Should NEVER happen, but good style to aid debugging
      error_message ( "Internal error in LayerNet::trial" ) ;
      return ;
      }

   if (nhid1 == 0) {                // No hidden layer
      switch (model) {
         case NETMOD_REAL:
            for (i=0 ; i<nout ; i++)
               activity_rr ( input , out_coefs+i*nin_n , out+i , nin , outlin );
            break ;
         case NETMOD_COMPLEX:
            for (i=0 ; i<nout ; i++)
               activity_cc ( input , out_coefs+i*nin_n , out+2*i , nin, outlin);
            break ;
         case NETMOD_COMPLEX_INPUT:
            for (i=0 ; i<nout ; i++)
               activity_cr ( input , out_coefs+i*nin_n , out+i , nin , outlin );
            break ;
         } // Switch on model
      }

   else if (nhid2 == 0) {           // One hidden layer
      switch (model) {
         case NETMOD_REAL:
            for (i=0 ; i<nhid1 ; i++)
               activity_rr ( input , hid1_coefs+i*nin_n , hid1+i , nin , 0 ) ;
            for (i=0 ; i<nout ; i++)
               activity_rr ( hid1 , out_coefs+i*nhid1_n , out+i, nhid1, outlin);
            break ;
         case NETMOD_COMPLEX:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , nin , 0 ) ;
            for (i=0 ; i<nout ; i++)
               activity_cc ( hid1, out_coefs+i*nhid1_n, out+2*i, nhid1, outlin);
            break ;
         case NETMOD_COMPLEX_INPUT:
            for (i=0 ; i<nhid1 ; i++)
               activity_cr ( input , hid1_coefs+i*nin_n , hid1+i , nin , 0 ) ;
            for (i=0 ; i<nout ; i++)
               activity_rr ( hid1 , out_coefs+i*nhid1_n , out+i, nhid1, outlin);
            break ;
         case NETMOD_COMPLEX_HIDDEN:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , nin , 0 ) ;
            for (i=0 ; i<nout ; i++)
               activity_cr ( hid1 , out_coefs+i*nhid1_n , out+i, nhid1, outlin);
            break ;
         } // Switch on model
      }

   else {                           // Two hidden layers
      switch (model) {
         case NETMOD_REAL:
            for (i=0 ; i<nhid1 ; i++)
               activity_rr ( input , hid1_coefs+i*nin_n , hid1+i , nin , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_rr ( hid1 , hid2_coefs+i*nhid1_n , hid2+i , nhid1 , 0 );
            for (i=0 ; i<nout ; i++)
               activity_rr ( hid2 , out_coefs+i*nhid2_n , out+i, nhid2, outlin);
            break ;
         case NETMOD_COMPLEX:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , nin , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_cc ( hid1 , hid2_coefs+i*nhid1_n , hid2+2*i , nhid1, 0);
            for (i=0 ; i<nout ; i++)
               activity_cc ( hid2, out_coefs+i*nhid2_n, out+2*i, nhid2, outlin);
            break ;
         case NETMOD_COMPLEX_INPUT:
            for (i=0 ; i<nhid1 ; i++)
               activity_cr ( input , hid1_coefs+i*nin_n , hid1+i , nin , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_rr ( hid1 , hid2_coefs+i*nhid1_n , hid2+i , nhid1 , 0 );
            for (i=0 ; i<nout ; i++)
               activity_rr ( hid2 , out_coefs+i*nhid2_n , out+i, nhid2, outlin);
            break ;
         case NETMOD_COMPLEX_HIDDEN:
            for (i=0 ; i<nhid1 ; i++)
               activity_cc ( input , hid1_coefs+i*nin_n , hid1+2*i , nin , 0 ) ;
            for (i=0 ; i<nhid2 ; i++)
               activity_cc ( hid1 , hid2_coefs+i*nhid1_n , hid2+2*i , nhid1, 0);
            for (i=0 ; i<nout ; i++)
               activity_cr ( hid2 , out_coefs+i*nhid2_n , out+i, nhid2, outlin);
            break ;
         } // Switch on model
      }
}


/*
--------------------------------------------------------------------------------

   trial_error - Compute the mean square error for the entire training set

--------------------------------------------------------------------------------
*/

double LayerNet::trial_error ( TrainingSet *tptr )
{
   int i, size, tset, tclass ;
   double err, tot_err, *dptr, diff, dsq, prev, denom, t, x, xx ;

   if (outmod == OUTMOD_CLASSIFY)   // Compute size of each training sample
      size = tptr->nin + 1 ;
   else if (outmod == OUTMOD_AUTO)
      size = tptr->nin ;
   else if (outmod == OUTMOD_GENERAL)
      size = tptr->nin + tptr->nout ;

   tot_err = 0.0 ;  // Total error will be cumulated here

   for (tset=0 ; tset<tptr->ntrain ; tset++) {  // Do all samples

      dptr = tptr->data + size * tset ;     // Point to this sample
      trial ( dptr ) ;                      // Evaluate network for it
      err = 0.0 ;

      if (outmod == OUTMOD_AUTO) {          // If this is AUTOASSOCIATIVE
         for (i=0 ; i<nout_n ; i++) {       // then the expected outputs
            diff = *dptr++ - out[i] ;       // are just the inputs
            switch (errtype) {
               case ERRTYPE_MSE:            // Sum for both types
               case ERRTYPE_LOGMEAN:
                  dsq = diff * diff ;
                  err += dsq ;
                  break ;
               case ERRTYPE_ABS:
                  if (model == NETMOD_COMPLEX) { // Sqrt ( r*r + i*i )
                     dsq = diff * diff ;
                     if (i % 2)
                        err += sqrt ( dsq + prev ) ;
                     else
                        prev = dsq ;
                     }
                  else
                     err += fabs(diff) ;
                  break ;
               case ERRTYPE_KK:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) {
                     if (i % 2) {
                        denom = 1.0 - out[i] * out[i] - out[i-1] * out[i-1] ;
                        if (denom < 1.e-10) // SHOULD never be negative
                           denom = 1.e-10 ;
                        err += (dsq + prev) / denom ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     denom = 1.0 - out[i] * out[i] ; // SHOULD never be negative
                     if (denom < 1.e-10)
                        denom = 1.e-10 ;
                     err += dsq / denom ;
                     }
                  break ;
               case ERRTYPE_MAX:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) {
                     if (i % 2) {
                        dsq += prev ;
                        if (dsq > err)
                           err = dsq ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     if (dsq > err)
                        err = dsq ;
                     }
                  break ;
               case ERRTYPE_SIXTEENTH:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) {
                     if (i % 2) {
                        dsq += prev ;
                        dsq = dsq * dsq ;
                        dsq = dsq * dsq ;
                        err += dsq * dsq ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     dsq = dsq * dsq ;
                     dsq = dsq * dsq ;
                     err += dsq * dsq ;
                     }
                  break ;
               case ERRTYPE_MEANLOG:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) { // Log of r*r + i*i
                     if (i % 2)
                        err += log ( dsq + prev + LOG_EPS ) ;
                     else
                        prev = dsq ;
                     }
                  else
                     err += log ( dsq + LOG_EPS ) ;
                  break ;
               } // Switch errtype
            } // For all outputs
         } // If OUTMOD_AUTO

      else if (outmod == OUTMOD_CLASSIFY) { // If this is Classification
         tclass = (int) dptr[tptr->nin] - 1 ; // class is stored after inputs
         for (i=0 ; i<nout ; i++) {         // Recall that train added a
            if (tclass == i)                // fraction so that the above
               t = NEURON_ON ;              // truncation to get tclass is
            else                            // always safe in any radix
               t = NEURON_OFF ;
            diff = t - out[i] ;
            switch (errtype) {
               case ERRTYPE_MSE:            // Sum for both types
               case ERRTYPE_LOGMEAN:
                  dsq = diff * diff ;
                  err += dsq ;
                  break ;
               case ERRTYPE_ABS:
                  err += fabs(diff) ;
                  break ;
               case ERRTYPE_KK:
                  dsq = diff * diff ;
                  denom = 1.0 - out[i] * out[i] ; // SHOULD never be negative
                  if (denom < 1.e-10)
                     denom = 1.e-10 ;
                  err += dsq / denom ;
                  break ;
               case ERRTYPE_XENT:
                  xx = out[i] ;
                  if (xx < -.99999999)
                     xx = -.99999999 ;
                  if (xx > .99999999)
                     xx = .99999999 ;
                  t = 0.5 * (t + 1.0) ;
                  x = 0.5 * (xx + 1.0) ;
                  err += t * log ( t/x ) + (1.0 - t) * log ((1.0-t) / (1.0-x));
                  break ;
               case ERRTYPE_MAX:
                  dsq = diff * diff ;
                  if (dsq > err)
                     err = dsq ;
                  break ;
               case ERRTYPE_SIXTEENTH:
                  dsq = diff * diff ;
                  dsq = dsq * dsq ;
                  dsq = dsq * dsq ;
                  err += dsq * dsq ;
                  break ;
               case ERRTYPE_MEANLOG:
                  dsq = diff * diff ;
                  err += log ( dsq + LOG_EPS ) ;
                  break ;
               } // Switch errtype
            } // For all outputs
         } // If OUTMOD_CLASSIFY

      else if (outmod == OUTMOD_GENERAL) {  // If this is GENERAL output
         dptr += tptr->nin ;                // outputs stored after inputs
         for (i=0 ; i<nout_n ; i++) {
            diff = *dptr++ - out[i] ;
            switch (errtype) {
               case ERRTYPE_MSE:            // Sum for both types
               case ERRTYPE_LOGMEAN:
                  dsq = diff * diff ;
                  err += dsq ;
                  break ;
               case ERRTYPE_ABS:
                  if (model == NETMOD_COMPLEX) { // Sqrt ( r*r + i*i )
                     dsq = diff * diff ;
                     if (i % 2)
                        err += sqrt ( dsq + prev ) ;
                     else
                        prev = dsq ;
                     }
                  else
                     err += fabs(diff) ;
                  break ;
               case ERRTYPE_KK:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) {
                     if (i % 2) {
                        denom = 1.0 - out[i] * out[i] - out[i-1] * out[i-1] ;
                        if (denom < 1.e-10) // SHOULD never be negative
                           denom = 1.e-10 ;
                        err += (dsq + prev) / denom ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     denom = 1.0 - out[i] * out[i] ; // SHOULD never be negative
                     if (denom < 1.e-10)
                        denom = 1.e-10 ;
                     err += dsq / denom ;
                     }
                  break ;
               case ERRTYPE_MAX:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) {
                     if (i % 2) {
                        dsq += prev ;
                        if (dsq > err)
                           err = dsq ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     if (dsq > err)
                        err = dsq ;
                     }
                  break ;
               case ERRTYPE_SIXTEENTH:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) {
                     if (i % 2) {
                        dsq += prev ;
                        dsq = dsq * dsq ;
                        dsq = dsq * dsq ;
                        err += dsq * dsq ;
                        }
                     else
                        prev = dsq ;
                     }
                  else {
                     dsq = dsq * dsq ;
                     dsq = dsq * dsq ;
                     err += dsq * dsq ;
                     }
                  break ;
               case ERRTYPE_MEANLOG:
                  dsq = diff * diff ;
                  if (model == NETMOD_COMPLEX) { // Log of r*r + i*i
                     if (i % 2)
                        err += log ( dsq + prev + LOG_EPS ) ;
                     else
                        prev = dsq ;
                     }
                  else
                     err += log ( dsq + LOG_EPS ) ;
                  break ;
               } // Switch errtype
            } // For all outputs
         } // If OUTMOD_GENERAL

      if (errtype == ERRTYPE_LOGMEAN)
         err = log ( err / (double) nout + LOG_EPS ) ;

      tot_err += err ;
      } // for all tsets

/*
   Find the mean per presentation.  Also, compensate for nout if that was
   not done implicitly in the error computation.  Finally, scale in such a
   way that the result is 0-100, for user's convenience.
*/

   neterr = tot_err / (double) tptr->ntrain ; // Mean per presentation

   switch (errtype) {
      case ERRTYPE_MSE:  // Max err per neuron is 2 squared
         neterr *= 25. / (double) nout ;
         break ;
      case ERRTYPE_ABS:  // Max err per neuron is 2
         neterr = 50. * neterr / (double) nout ;
         break ;
      case ERRTYPE_KK:
         neterr = 25. * neterr / (double) nout ;
         break ;
      case ERRTYPE_XENT:
         neterr = 37.7 * neterr / (double) nout ;
         break ;
      case ERRTYPE_MAX:  // Max err is 4
         neterr *= 25. ;
         break ;
      case ERRTYPE_SIXTEENTH:  // Max err per neuron is 2 ** 16
         neterr *= 1000.0 / (double) nout ;
         break ;
      case ERRTYPE_LOGMEAN:
         neterr = (neterr + EPS_OFFSET) * 100. / LOG_FACTOR ;
         break ;
      case ERRTYPE_MEANLOG:  // Err per neuron is -EPS_OFFSET to log(2 squared)
         neterr = (neterr / (double) nout + EPS_OFFSET) * 100. / LOG_FACTOR ;
         break ;
      }
   return neterr ;
}


/*
--------------------------------------------------------------------------------

   learn

--------------------------------------------------------------------------------
*/

void LayerNet::learn ( TrainingSet *tptr , struct LearnParams *lptr )
{
   if (! exe) {   // Should NEVER happen, but good style to aid debugging
      error_message ( "Internal error in LayerNet::learn" ) ;
      return ;
      }

   this->errtype = lptr->errtype ;  // Tell net routines what our error def is

   if (lptr->method == METHOD_AN1)
      an1 ( tptr , lptr ) ;
   else if (lptr->method == METHOD_AN1_CJ)
      an1_cj ( tptr , lptr ) ;
   else if (lptr->method == METHOD_REGRS_CJ)
      regrs_cj ( tptr , lptr ) ;

   return ;
}


/*
--------------------------------------------------------------------------------

   wt_print - Print weights as ASCII to file
   wt_save - Save weights to disk (called from WT_SAVE.CPP)
   wt_restore - Restore weights from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

void LayerNet::wt_print ( char *name )
{
   int i, j, k ;
   char msg[81] ;
   FILE *fp ;

   if ((fp = fopen ( name , "wt" )) == NULL) {
      strcpy ( msg , "Cannot open WEIGHT PRINT file " ) ;
      strcat ( msg , name ) ;
      error_message ( msg ) ;
      return ;
      }

   fprintf ( fp , "MLFN ASCII weight file" ) ;

   if (nhid1 == 0) {                // No hidden layer
      k = 0 ;
      for (i=0 ; i<nout ; i++) {
         fprintf ( fp , "\nInput-to-Output neuron %d weights, bias last:", i+1);
         if (model == NETMOD_REAL) {
            for (j=0 ; j<=nin ; j++)
               fprintf ( fp , "\n%lf", out_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nin ; j++) {
               fprintf ( fp , "\n(%lf %lf)", out_coefs[k], out_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      }

   else if (nhid2 == 0) {           // One hidden layer
      k = 0 ;
      for (i=0 ; i<nhid1 ; i++) {
         fprintf ( fp , "\nInput-to-Hid1 neuron %d weights, bias last:", i+1 ) ;
         if (model == NETMOD_REAL) {
            for (j=0 ; j<=nin ; j++)
               fprintf ( fp , "\n%lf", hid1_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nin ; j++) {
               fprintf ( fp , "\n(%lf %lf)", hid1_coefs[k], hid1_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      k = 0 ;
      for (i=0 ; i<nout ; i++) {
         fprintf ( fp , "\nHid1-to-Output neuron %d weights, bias last:", i+1 );
         if ((model == NETMOD_REAL)  ||  (model == NETMOD_COMPLEX_INPUT)) {
            for (j=0 ; j<=nhid1 ; j++)
               fprintf ( fp , "\n%lf", out_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nhid1 ; j++) {
               fprintf ( fp , "\n(%lf %lf)", out_coefs[k], out_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      }

   else {                           // Two hidden layers
      k = 0 ;
      for (i=0 ; i<nhid1 ; i++) {
         fprintf ( fp , "\nInput-to-Hid1 neuron %d weights, bias last:", i+1 ) ;
         if (model == NETMOD_REAL) {
            for (j=0 ; j<=nin ; j++)
               fprintf ( fp , "\n%lf", hid1_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nin ; j++) {
               fprintf ( fp , "\n(%lf %lf)", hid1_coefs[k], hid1_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      k = 0 ;
      for (i=0 ; i<nhid2 ; i++) {
         fprintf ( fp , "\nHid1-to-Hid2 neuron %d weights, bias last:", i+1 ) ;
         if ((model == NETMOD_REAL)  ||  (model == NETMOD_COMPLEX_INPUT)) {
            for (j=0 ; j<=nhid1 ; j++)
               fprintf ( fp , "\n%lf", hid2_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nhid1 ; j++) {
               fprintf ( fp , "\n(%lf %lf)", hid2_coefs[k], hid2_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      k = 0 ;
      for (i=0 ; i<nout ; i++) {
         fprintf ( fp , "\nHid2-to-Output neuron %d weights, bias last:", i+1 );
         if ((model == NETMOD_REAL)  ||  (model == NETMOD_COMPLEX_INPUT)) {
            for (j=0 ; j<=nhid2 ; j++)
               fprintf ( fp , "\n%lf", out_coefs[k++] ) ;
            }
         else {
            for (j=0 ; j<=nhid2 ; j++) {
               fprintf ( fp , "\n(%lf %lf)", out_coefs[k], out_coefs[k+1] ) ;
               k += 2 ;
               }
            }
         }
      }

   fclose ( fp ) ;

}

int LayerNet::wt_save ( FILE *fp )
{
   int n ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      fwrite ( out_coefs , n * sizeof(double) , 1 , fp ) ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      fwrite ( hid1_coefs , n * sizeof(double) , 1 , fp ) ;
      n = nout * nhid1_n ;
      fwrite ( out_coefs , n * sizeof(double) , 1 , fp ) ;
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      fwrite ( hid1_coefs , n * sizeof(double) , 1 , fp ) ;
      n = nhid2 * nhid1_n ;
      fwrite ( hid2_coefs , n * sizeof(double) , 1 , fp ) ;
      n = nout * nhid2_n ;
      fwrite ( out_coefs , n * sizeof(double) , 1 , fp ) ;
      }

   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

void LayerNet::wt_restore ( FILE *fp )
{
   int n ;

   if (nhid1 == 0) {                // No hidden layer
      n = nout * nin_n ;
      fread ( out_coefs , n * sizeof(double) , 1 , fp ) ;
      }

   else if (nhid2 == 0) {           // One hidden layer
      n = nhid1 * nin_n ;
      fread ( hid1_coefs , n * sizeof(double) , 1 , fp ) ;
      n = nout * nhid1_n ;
      fread ( out_coefs , n * sizeof(double) , 1 , fp ) ;
      }

   else {                           // Two hidden layers
      n = nhid1 * nin_n ;
      fread ( hid1_coefs , n * sizeof(double) , 1 , fp ) ;
      n = nhid2 * nhid1_n ;
      fread ( hid2_coefs , n * sizeof(double) , 1 , fp ) ;
      n = nout * nhid2_n ;
      fread ( out_coefs , n * sizeof(double) , 1 , fp ) ;
      }

   if (ferror ( fp ))
      ok = 0 ;
}
