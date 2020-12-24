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
/*  gradient - Called by CONJGRAD to compute weight gradient                  */
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

double LayerNet::gradient (
   TrainingSet *tptr ,
   double *work1 ,
   double *work2 ,
   double *grad
   )
{
   if (model == NETMOD_REAL)
      return gradient_real ( tptr , work1 , work2 , grad ) ;
   else 
      return gradient_complex ( tptr , work1 , work2 , grad ) ;
}

/*
--------------------------------------------------------------------------------

   GRADIENT_REAL - Gradient for pure real model

--------------------------------------------------------------------------------
*/

double LayerNet::gradient_real (
   TrainingSet *tptr ,
   double *hid2delta ,
   double *outdelta ,
   double *grad
   )
{
   int i, j, size, tset, tclass, n, nprev, nnext ;
   double err, error, *dptr, delta, *hid1grad, *hid2grad, *outgrad ;
   double *outprev, *prevact, *nextcoefs, *nextdelta, *gradptr, factor, targ ;

/*
   Compute size of each training sample
*/

   if (outmod == OUTMOD_CLASSIFY)
      size = tptr->nin + 1 ;
   else if (outmod == OUTMOD_AUTO)
      size = tptr->nin ;
   else if (outmod == OUTMOD_GENERAL)
      size = tptr->nin + tptr->nout ;

/*
   Compute length of grad vector and gradient positions in it.
   Also point to layer previous to output and its size.
   Ditto for layer after hid1.
   For nprev and nnext, we use the _w version because they refer to actual
   neuron activations.  Bias is handled separately.
*/

   if (nhid1 == 0) {      // No hidden layer
      n = nout * nin_n ;
      outgrad = grad ;
      nprev = nin_w ;
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + nout * nhid1_n ;
      hid1grad = grad ;
      outgrad = grad + nhid1 * nin_n ;
      outprev = hid1 ;
      nprev = nhid1_w ;
      nnext = nout_n ;
      nextcoefs = out_coefs ;
      nextdelta = outdelta ;
      }
   else {                 // Two hidden layers
      n = nhid1 * nin_n + nhid2 * nhid1_n + nout * nhid2_n ;
      hid1grad = grad ;
      hid2grad = grad + nhid1 * nin_n ;
      outgrad = hid2grad + nhid2 * nhid1_n ;
      outprev = hid2 ;
      nprev = nhid2_w ;
      nnext = nhid2_w ;
      nextcoefs = hid2_coefs ;
      nextdelta = hid2delta ;
      }

   for (i=0 ; i<n ; i++)  // Zero gradient for summing
      grad[i] = 0.0 ;

   error = 0.0 ;  // Will cumulate total error here
   for (tset=0 ; tset<tptr->ntrain ; tset++) { // Do all samples

      dptr = tptr->data + size * tset ;     // Point to this sample
      trial ( dptr ) ;                      // Evaluate network for it
      err = 0.0 ;                           // Cumulates for this presentation

      if (outmod == OUTMOD_AUTO) {          // If this is AUTOASSOCIATIVE
         for (i=0 ; i<nout ; i++)           // then targets are inputs
            errderiv_r ( nout , i , out , dptr[i] , &err , outdelta ) ;
         }

      else if (outmod == OUTMOD_CLASSIFY) { // If this is Classification
         tclass = (int) dptr[tptr->nin] - 1 ; // class is stored after inputs
         for (i=0 ; i<nout ; i++) {         // Recall that train added a
            if (tclass == i)                // fraction so that the above
               targ = NEURON_ON ;           // truncation to get tclass is
            else                            // always safe in any radix
               targ = NEURON_OFF ;         
            errderiv_r ( nout , i , out , targ , &err , outdelta ) ;
            }
         }

      else if (outmod == OUTMOD_GENERAL) {  // If this is GENERAL output
         dptr += tptr->nin ;                // outputs stored after inputs
         for (i=0 ; i<nout ; i++)
            errderiv_r ( nout , i , out , dptr[i] , &err , outdelta ) ;
         }

      if (! outlin) {
         for (i=0 ; i<nout ; i++)
            outdelta[i] *= actderiv ( out[i] ) ;
         }

      error += err ;                        // Cumulate presentation into epoch 

/*
   Cumulate output gradient
*/

      if (nhid1 == 0)         // No hidden layer
         prevact = tptr->data + size * tset ;
      else
         prevact = outprev ;  // Point to previous layer
      gradptr = outgrad ;
      for (i=0 ; i<nout ; i++) {
         delta = outdelta[i] ;
         for (j=0 ; j<nprev ; j++)
            *gradptr++ += delta * prevact[j] ;
         *gradptr++ += delta ;   // Bias activation is always 1
         }

/*
   Cumulate hid2 gradient (if it exists)
*/
   
      if (nhid2) {
         gradptr = hid2grad ;
         for (i=0 ; i<nhid2 ; i++) {
            delta = 0.0 ;
            for (j=0 ; j<nout ; j++)
               delta += outdelta[j] * out_coefs[j*(nhid2+1)+i] ;
            delta *= actderiv ( hid2[i] ) ;
            hid2delta[i] = delta ;
            for (j=0 ; j<nhid1 ; j++)
               *gradptr++ += delta * hid1[j] ;
            *gradptr++ += delta ;   // Bias activation is always 1
            // Additional bias stuff here
            }
         }

/*
   Cumulate hid1 gradient (if it exists)
*/
   
      if (nhid1) {
         prevact = tptr->data + size * tset ;
         gradptr = hid1grad ;
         for (i=0 ; i<nhid1 ; i++) {
            delta = 0.0 ;
            for (j=0 ; j<nnext ; j++)
               delta += nextdelta[j] * nextcoefs[j*(nhid1+1)+i] ;
            delta *= actderiv ( hid1[i] ) ;
            for (j=0 ; j<nin ; j++)
               *gradptr++ += delta * prevact[j] ;
            *gradptr++ += delta ;   // Bias activation is always 1
            // Additional bias stuff here
            }
         }

      } // for all tsets
   
/*
   Find the mean per presentation.  Also, compensate for nout if that was
   not done implicitly in the error computation.  Finally, scale in such a
   way that the result is 0-100, for user's convenience.
   Remember that if we multiply the error, we must do the same to the gradient.
*/

   factor = 1.0 / (double) tptr->ntrain ;

   switch (errtype) {
      case ERRTYPE_MSE:
         factor *= 25. / (double) nout ; // Max err per output is 2 squared
         error *= factor ;
         break ;
      case ERRTYPE_ABS:  // Max err per neuron is 2
         factor *= 50. / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_KK:
         factor *= 25.0 / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_XENT:
         factor *= 37.7 / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_MAX:
         factor *= 25. ; // Max err 2 squared
         error *= factor ;
         break ;
      case ERRTYPE_SIXTEENTH:  // Max err per neuron is 2 ** 16
         factor *= 1000.0 / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_LOGMEAN:
         error *= factor ;
         error = (error + EPS_OFFSET) * 100. / LOG_FACTOR ;
         factor *= 100. / LOG_FACTOR ;
         break ;
      case ERRTYPE_MEANLOG:  // Err per neuron is -EPS_OFFSET to log(2 squared)
         error *= factor ;
         error = (error / (double) nout + EPS_OFFSET) * 100. / LOG_FACTOR ;
         factor = factor / (double) nout * 100. / LOG_FACTOR ;
         break ;
      }

   for (i=0 ; i<n ; i++)
      grad[i] *= factor ;

   return error ;
}


/*
--------------------------------------------------------------------------------

   GRADIENT_COMPLEX - Gradient for all complex models

--------------------------------------------------------------------------------
*/

double LayerNet::gradient_complex (
   TrainingSet *tptr ,
   double *work1 ,
   double *work2 ,
   double *grad
   )
{
   int i, j, k, size, tset, tclass, n, nprev ;
   double err, error, *dptr, *prevact, *hid1grad, *outgrad, *gradptr ;
   double rdiff, idiff, rsum, isum ;
   double factor, rdelta, idelta, targ ;
   double *dar10 ;   // Partial of real attained output wrt real net
   double *dai10 ;   // Partial of imaginary attained output wrt real net
   double *dar01 ;   // Partial of real attained output wrt imaginary net
   double *dai01 ;   // Partial of imaginary attained output wrt imaginary net
   double *dhr10 ;   // Partial of real hidden wrt real net
   double *dhi10 ;   // Partial of imaginary hidden wrt real net
   double *dhr01 ;   // Partial of real hidden wrt imaginary net
   double *dhi01 ;   // Partial of imaginary hidden wrt imaginary net
   double *hend  ;   // Points to next byte after dh's
   double *delta_r ; // Real part of delta
   double *delta_i ; // and imaginary part

/*
   Compute size of each training sample
*/

   if (outmod == OUTMOD_CLASSIFY)
      size = tptr->nin + 1 ;
   else if (outmod == OUTMOD_AUTO)
      size = tptr->nin ;
   else if (outmod == OUTMOD_GENERAL)
      size = tptr->nin + tptr->nout ;

/*
   Compute length of grad vector and gradient positions in it.
   Also compute positions in work1 where the various partial derivatives
   of the output activation are stored.
*/

   if (nhid1 == 0) {      // No hidden layer
      n = nout * nin_n ;
      outgrad = grad ;
      dar10 = work1 ;
      dar01 = dar10 + nout ;
      if (model == NETMOD_COMPLEX) {
         dai10 = dar01 + nout ;
         dai01 = dai10 + nout ;
         }
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + nout * nhid1_n ;
      hid1grad = grad ;
      outgrad = hid1grad + nhid1 * nin_n ;
      dhr10 = work1 ;
      dhr01 = dhr10 + nhid1 ;
      if (model == NETMOD_COMPLEX_INPUT)
         hend = dhr01 + nhid1 ;
      else {
         dhi10 = dhr01 + nhid1 ;
         dhi01 = dhi10 + nhid1 ;
         hend = dhi01 + nhid1 ;
         }
      dar10 = hend ;
      delta_r = dar10 + nout ;
      if (model != NETMOD_COMPLEX_INPUT) {
         dar01 = delta_r + nout ;
         delta_i = dar01 + nout ;
         if (model == NETMOD_COMPLEX) {
            dai10 = delta_i + nout ;
            dai01 = dai10 + nout ;
            }
         }
      }

/*
   This is the main loop which cumulates across the epoch
*/
   error = 0.0 ;           // Will cumulate total error here
   for (i=0 ; i<n ; i++)   // Zero gradient for summing
      grad[i] = 0.0 ;
   for (tset=0 ; tset<tptr->ntrain ; tset++) { // Do all samples

      dptr = tptr->data + size * tset ;     // Point to this sample
      err = 0.0 ;

/*
   Compute all activations and partial derivatives of activations
*/

      if (nhid1 == 0) {                // No hidden layer
         switch (model) {
            case NETMOD_COMPLEX_INPUT:
               for (i=0 ; i<nout ; i++)
                  partial_cr ( dptr , out_coefs+i*nin_n , out+i , nin ,
                               dar10+i , dar01+i , outlin ) ;
               break ;
            case NETMOD_COMPLEX:
               for (i=0 ; i<nout ; i++)
                  partial_cc ( dptr , out_coefs+i*nin_n , out+2*i , nin ,
                               dar10+i , dar01+i , dai10+i , dai01+i , outlin );
               break ;
            } // Switch on model
         }

      else if (nhid2 == 0) {           // One hidden layer
         switch (model) {
            case NETMOD_COMPLEX_INPUT:
               for (i=0 ; i<nhid1 ; i++)
                  partial_cr ( dptr , hid1_coefs+i*nin_n , hid1+i , nin ,
                               dhr10+i , dhr01+i , 0 ) ;
               for (i=0 ; i<nout ; i++) {
                  activity_rr ( hid1 , out_coefs+i*nhid1_n , out+i , nhid1 ,
                                outlin ) ;
                  if (outlin)
                     dar10[i] = 1.0 ;
                  else 
                     dar10[i] = actderiv ( out[i] ) ;
                  }
               break ;
            case NETMOD_COMPLEX_HIDDEN:
               for (i=0 ; i<nhid1 ; i++)
                  partial_cc ( dptr , hid1_coefs+i*nin_n , hid1+2*i , nin ,
                               dhr10+i , dhr01+i , dhi10+i , dhi01+i , 0 ) ;
               for (i=0 ; i<nout ; i++)
                  partial_cr ( hid1 , out_coefs+i*nhid1_n , out+i , nhid1 ,
                               dar10+i , dar01+i , outlin ) ;
               break ;
            case NETMOD_COMPLEX:
               for (i=0 ; i<nhid1 ; i++)
                  partial_cc ( dptr , hid1_coefs+i*nin_n , hid1+2*i , nin ,
                               dhr10+i , dhr01+i , dhi10+i , dhi01+i , 0 ) ;
               for (i=0 ; i<nout ; i++)
                  partial_cc ( hid1 , out_coefs+i*nhid1_n , out+2*i , nhid1 ,
                               dar10+i , dar01+i , dai10+i , dai01+i , outlin );
               break ;
            } // Switch on model
         }

/*
   Compute output error and its derivative with respect to output activation.
   (Put derivative in work2)
*/

      if (outmod == OUTMOD_AUTO) {
         if (model == NETMOD_COMPLEX)
            for (i=0 ; i<nout ; i++)
               errderiv_c ( nout , i , out , dptr[2*i] , dptr[2*i+1] ,
                            &err , work2 ) ;
         else 
            for (i=0 ; i<nout ; i++)
               errderiv_r ( nout , i , out , dptr[i] , &err , work2 ) ;
         }

      else if (outmod == OUTMOD_CLASSIFY) { // If this is Classification
         tclass = (int) dptr[tptr->nin] - 1 ; // class is stored after inputs
         for (i=0 ; i<nout ; i++) {         // Recall that train added a
            if (tclass == i)                // fraction so that the above
               targ = NEURON_ON          ;  // truncation to get tclass is
            else                            // always safe in any radix
               targ = NEURON_OFF ;
            errderiv_r ( nout , i , out , targ , &err , work2 ) ;
            }
         }

      else if (outmod == OUTMOD_GENERAL) {  // If this is GENERAL output
         dptr += tptr->nin ;                // outputs stored after inputs
         if (model == NETMOD_COMPLEX)
            for (i=0 ; i<nout ; i++)
               errderiv_c ( nout , i , out , dptr[2*i] , dptr[2*i+1] ,
                            &err , work2 ) ;
         else
            for (i=0 ; i<nout ; i++)
               errderiv_r ( nout , i , out , dptr[i] , &err , work2 ) ;
         }

      error += err ;                        // Cumulate presentation into epoch

/*
   Cumulate output gradient.  Prevact is the activity in the layer
   just prior to the output layer.
*/

      if (nhid1 == 0) {        // No hidden layer
         nprev = nin ;
         prevact = tptr->data + size * tset ;
         }
      else {
         nprev = nhid1 ;
         prevact = hid1 ;
         }

      gradptr = outgrad ;

      if ((model == NETMOD_COMPLEX_INPUT)  &&  nhid1) {  // Real-to-real
         for (i=0 ; i<nout ; i++) {     // Do every output neuron
            rdiff = work2[i] ;          // Target minus attained output
            rdelta = rdiff * dar10[i] ;
            delta_r[i] = rdelta ;       // Save for hidden layer computations
            for (j=0 ; j<nprev ; j++)   // Connection to every previous neuron
               *gradptr++ += rdelta * prevact[j] ;
            *gradptr++ += rdelta ;  // Bias (prevact=1)
            }
         }

      else {                            // Complex domain
         for (i=0 ; i<nout ; i++) {     // Do every output neuron
            if (model == NETMOD_COMPLEX) { // Complex range
               rdiff = work2[2*i] ;     // Target minus attained output
               idiff = work2[2*i+1] ;   // And its imaginary part
               rdelta = rdiff * dar10[i]  +  idiff * dai10[i] ;
               idelta = rdiff * dar01[i]  +  idiff * dai01[i] ;
               }
            else {
               rdiff = work2[i] ;          // Target minus attained output
               rdelta = rdiff * dar10[i] ;
               idelta = rdiff * dar01[i] ;
               }
            if (nhid1) {                // Save for hidden layer grad
               delta_r[i] = rdelta ;
               delta_i[i] = idelta ;
               }
            for (j=0 ; j<nprev ; j++) { // Connection to every previous neuron
               *gradptr++ += rdelta * prevact[2*j] + idelta * prevact[2*j+1] ;
               *gradptr++ += -rdelta * prevact[2*j+1] + idelta * prevact[2*j] ;
               }  // Next two lines are bias (prevact = 1 + 0 i)
            *gradptr++ += rdelta ;
            *gradptr++ += idelta ;
            }
         }

/*
   Cumulate hid1 gradient (if it exists)
*/
   
      if (nhid1) {
         prevact = tptr->data + size * tset ;
         gradptr = hid1grad ;

         for (i=0 ; i<nhid1 ; i++) {    // For every hidden neuron

            // Sum out weight * out delta, then multiply by grad to get delta
            rsum = isum = 0.0 ;
            if (model == NETMOD_COMPLEX_INPUT) {  // Complex-to-real
               for (k=0 ; k<nout ; k++)    // Sum for all outputs
                  rsum += delta_r[k] * out_coefs[k*nhid1_n+i] ;
               rdelta = rsum * dhr10[i] ;
               idelta = rsum * dhr01[i] ;
               }
            else {  // Complex-to-complex
               for (k=0 ; k<nout ; k++) {  // Sum for all outputs
                  rsum += delta_r[k] * out_coefs[k*nhid1_n+2*i] +
                          delta_i[k] * out_coefs[k*nhid1_n+2*i+1] ;
                  isum += -delta_r[k] * out_coefs[k*nhid1_n+2*i+1] +
                          delta_i[k] * out_coefs[k*nhid1_n+2*i] ;
                  }
               rdelta = rsum * dhr10[i]  +  isum * dhi10[i] ;
               idelta = rsum * dhr01[i]  +  isum * dhi01[i] ;
               }

            // Gradient is delta times previous layer activity
            for (j=0 ; j<nin ; j++) {  // For every input connection
               *gradptr++ += rdelta * prevact[2*j] + idelta * prevact[2*j+1];
               *gradptr++ += -rdelta * prevact[2*j+1] + idelta* prevact[2*j];
               }  // Next two lines are bias (prevact = 1 + 0 i)
            *gradptr++ += rdelta ;
            *gradptr++ += idelta ;
            } // For every input connection
         } // If hidden layer

      } // for all tsets
   
/*
   Find the mean per presentation.  Also, compensate for nout if that was
   not done implicitly in the error computation.  Finally, scale in such a
   way that the result is 0-100, for user's convenience.
   Remember that if we multiply the error, we must do the same to the gradient.
*/

   factor = 1.0 / (double) tptr->ntrain ;

   switch (errtype) {
      case ERRTYPE_MSE:
         factor *= 25. / (double) nout ; // Max err per output is 2 squared
         error *= factor ;
         break ;
      case ERRTYPE_ABS:  // Max err per neuron is 2
         factor *= 50. / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_KK:
         factor *= 25.0 / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_XENT:
         factor *= 37.7 / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_MAX:
         factor *= 25. ; // Max err 2 squared
         error *= factor ;
         break ;
      case ERRTYPE_SIXTEENTH:  // Max err per neuron is 2 ** 16
         factor *= 1000.0 / (double) nout ;
         error *= factor ;
         break ;
      case ERRTYPE_LOGMEAN:
         error *= factor ;
         error = (error + EPS_OFFSET) * 100. / LOG_FACTOR ;
         factor *= 100. / LOG_FACTOR ;
         break ;
      case ERRTYPE_MEANLOG:  // Err per neuron is -EPS_OFFSET to log(2 squared)
         error *= factor ;
         error = (error / (double) nout + EPS_OFFSET) * 100. / LOG_FACTOR ;
         factor = factor / (double) nout * 100. / LOG_FACTOR ;
         break ;
      }

   for (i=0 ; i<n ; i++)
      grad[i] *= factor ;

   return error ;
}

/*
--------------------------------------------------------------------------------

   These two routines compute the (negative) derivative of the network error
   with respect to the output activation.  There is a real and complex version.

--------------------------------------------------------------------------------
*/

static int ibest ;  // For ERRTYPE_MAX

void LayerNet::errderiv_r (
   int nout ,        // Number of output neurons
   int iout ,        // Of which we are doing this one (org 0)
   double *outs ,    // All output activations
   double target ,   // Target for this one
   double *err ,     // Input of error so far, output cumulated per this
   double *deriv     // Output: deriv[iout] = deriv of err wrt outs[iout]
   )
{
   int i ;
   double d, dsq, denom, t, x, xx ;

   d = outs[iout] - target ;
   dsq = d * d ;

   switch (errtype) {

      case ERRTYPE_MSE:
         *err += dsq ;
         deriv[iout] = -2.0 * d ;
         return ;

      case ERRTYPE_ABS:
         *err += fabs ( d ) ;
         if (d < 0.0)
            deriv[iout] = 1.0 ;
         else if (d > 0.0)
            deriv[iout] = -1.0 ;
         else 
            deriv[iout] = 0.0 ;
         return ;

      case ERRTYPE_KK:
         denom = 1.0 - outs[iout] * outs[iout] ; // SHOULD never be negative
         if (denom < 1.e-10)
            denom = 1.e-10 ;
         *err += dsq / denom ;
         deriv[iout] = -2.0 / denom * (outs[iout] * dsq / denom + d) ;
         return ;

      case ERRTYPE_XENT:
         xx = outs[iout] ;
         if (xx < -.99999999)
            xx = -.99999999 ;
         if (xx > .99999999)
            xx = .99999999 ;
         t = 0.5 * (target + 1.0) ;
         x = 0.5 * (xx + 1.0) ;
         *err += t * log ( t/x ) + (1.0 - t) * log ((1.0-t) / (1.0-x));
         deriv[iout] = -0.5 *
                     ((1.0 - t) / (1.0 - x)  -  (1.0 + target) / (1.0 + xx )) ;
         return ;

      case ERRTYPE_MAX:
         if (dsq > *err) {
            *err = dsq ;
            ibest = iout ;
            }
         deriv[iout] = d ;
         if (iout < nout-1)
            return ;
         for (i=0 ; i<nout ; i++) {
            if (i == ibest)
               deriv[i] = -2.0 * deriv[i] ;
            else 
               deriv[i] = 0.0 ;
            }
         return ;

      case ERRTYPE_SIXTEENTH:
         dsq = dsq * dsq ;
         dsq = dsq * dsq ;
         dsq = dsq * dsq ;
         *err += dsq ;
         deriv[iout] = -16.0 * dsq / d ;
         return ;

      case ERRTYPE_LOGMEAN:
         *err += dsq ;
         deriv[iout] = d ;
         if (iout < nout-1)
            return ;
         for (i=0 ; i<nout ; i++)
            deriv[i] = -2.0 * deriv[i] / (*err + LOG_EPS * (double) nout ) ;
         *err = log ( *err / (double) nout + LOG_EPS ) ;
         return ;

      case ERRTYPE_MEANLOG:
         *err += log ( dsq  + LOG_EPS ) ;
         deriv[iout] = -2.0 * d / (dsq + LOG_EPS) ;
         return ;
      }
}

void LayerNet::errderiv_c (
   int nout ,        // Number of output neurons
   int iout ,        // Of which we are doing this one (org 0)
   double *outs ,    // All output activations (real, imaginary, ...)
   double target_r , // Target for this one
   double target_i , // And imaginary part
   double *err ,     // Input of error so far, output cumulated per this
   double *deriv     // Output: deriv[iout] = deriv of err wrt outs[iout]
   )
{
   int i ;
   double dr, di, dsq, p, denom ;

   dr = outs[2*iout] - target_r ;
   di = outs[2*iout+1] - target_i ;
   dsq = dr * dr  +  di * di ;

   switch (errtype) {

      case ERRTYPE_MSE:
         *err += dsq ;
         deriv[2*iout]   = -2.0 * dr ;
         deriv[2*iout+1] = -2.0 * di ;
         return ;

      case ERRTYPE_ABS:
         if (dsq == 0.0) {
            deriv[2*iout] = deriv[2*iout+1] = 0.0 ;
            return ;
            }
         p = sqrt ( dsq ) ;
         *err += p ;
         deriv[2*iout] = -dr / p ;
         deriv[2*iout+1] = -di / p ;
         return ;

      case ERRTYPE_KK:
         denom = 1.0 - outs[2*iout] * outs[2*iout]
                     - outs[2*iout+1] * outs[2*iout+1] ;
         if (denom < 1.e-10)  // SHOULD never be negative
            denom = 1.e-10 ;
         *err += dsq / denom ;
         deriv[2*iout]   = -2.0 / denom * (outs[2*iout]   * dsq / denom + dr) ;
         deriv[2*iout+1] = -2.0 / denom * (outs[2*iout+1] * dsq / denom + di) ;
         return ;

      case ERRTYPE_MAX:
         if (dsq > *err) {
            *err = dsq ;
            ibest = iout ;
            }
         deriv[2*iout]   = dr ;
         deriv[2*iout+1] = di ;
         if (iout < nout-1)
            return ;
         for (i=0 ; i<nout ; i++) {
            if (i == ibest) {
               deriv[2*i]   = -2.0 * deriv[2*i] ;
               deriv[2*i+1] = -2.0 * deriv[2*i+1] ;
               }
            else 
               deriv[2*i] = deriv[2*i+1] = 0.0 ;
            }
         return ;

      case ERRTYPE_SIXTEENTH:
         p = dsq ;
         dsq = dsq * dsq ;
         dsq = dsq * dsq ;
         dsq = dsq * dsq ;
         *err += dsq ;
         deriv[2*iout]   = -16.0 * dr * dsq / p ;
         deriv[2*iout+1] = -16.0 * di * dsq / p ;
         return ;

      case ERRTYPE_LOGMEAN:
         *err += dsq ;
         deriv[2*iout]   = dr ;
         deriv[2*iout+1] = di ;
         if (iout < nout-1)
            return ;
         denom = *err + LOG_EPS*(double) nout ;
         for (i=0 ; i<nout ; i++) {
            deriv[2*i]   = -2.0 * deriv[2*i]   / denom ;
            deriv[2*i+1] = -2.0 * deriv[2*i+1] / denom ;
            }
         *err = log ( *err / (double) nout + LOG_EPS ) ;
         return ;

      case ERRTYPE_MEANLOG:
         *err += log ( dsq  + LOG_EPS ) ;
         deriv[2*iout]   = -2.0 * dr / (dsq + LOG_EPS) ;
         deriv[2*iout+1] = -2.0 * di / (dsq + LOG_EPS) ;
         return ;
      }
}
