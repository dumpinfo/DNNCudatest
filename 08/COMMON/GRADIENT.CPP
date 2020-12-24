/******************************************************************************/
/*                                                                            */
/*  gradient - Called by CONJGRAD to compute weight gradient                  */
/*             Also called by SSG.                                            */
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

#define DEBUG 0

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

double MLFN::gradient (
   TrainingSet *tptr ,
   double *work1 ,
   double *work2 ,
   double *grad
   )
{
   if (domain == DOMAIN_REAL)
      return gradient_real ( tptr , work1 , work2 , grad ) ;
   else 
      return gradient_complex ( tptr , work1 , work2 , grad ) ;
}

/*
--------------------------------------------------------------------------------

   GRADIENT_REAL - Gradient for pure real model

--------------------------------------------------------------------------------
*/

double MLFN::gradient_real (
   TrainingSet *tptr ,
   double *delta2 ,
   double *delta_out ,
   double *grad
   )
{
   int i, j, size, casenum, true_class, n, n_before, n_after ;
   double target, err, error, *grad1, *grad2, *grad_out, *inptr, delta ;
   double *gradient, factor, *before_out, *act_before ;
   double neuron_on, neuron_off, *w_after, *delta_after ;

   if (outlin  &&  (errtype != ERRTYPE_XENT)  &&  (errtype != ERRTYPE_KK)) {
      neuron_on = NEURON_ON ;
      neuron_off = NEURON_OFF ;
      }
   else {
      neuron_on = 0.9 * NEURON_ON ;
      neuron_off = 0.9 * NEURON_OFF ;
      }

/*
   Compute size of each training set case
*/

   if (output_mode == OUTMOD_CLASSIFICATION)
      size = tptr->n_inputs + 1 ;
   else if (output_mode == OUTMOD_MAPPING)
      size = tptr->n_inputs + tptr->n_outputs ;

/*
   Compute lengths of vectors and the gradient positions in it.
   Also point to the layer just before the output.
   For n_before and n_after, we use the _w version because they refer to actual
   neuron activations.  Bias is handled separately.
*/

   if (nhid1 == 0) {      // No hidden layer
      n = n_outputs * nin_n ;
      n_before = nin_w ;
      grad_out = grad ;
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + n_outputs * nhid1_n ;
      n_before = nhid1_w ;
      n_after = nout_n ;
      grad1 = grad ;
      grad_out = grad + nhid1 * nin_n ;
      before_out = hid1 ;
      w_after = out_coefs ;
      delta_after = delta_out ;
      }
   else {                 // Two hidden layers
      n = nhid1 * nin_n + nhid2 * nhid1_n + n_outputs * nhid2_n ;
      n_before = nhid2_w ;
      n_after = nhid2_w ;
      grad1 = grad ;
      grad2 = grad + nhid1 * nin_n ;
      grad_out = grad2 + nhid2 * nhid1_n ;
      before_out = hid2 ;
      w_after = hid2_coefs ;
      delta_after = delta2 ;
      }

   for (i=0 ; i<n ; i++)  // Zero gradient; will sum here
      grad[i] = 0.0 ;

   error = 0.0 ;  // Epoch error cumulated here
   for (casenum=0 ; casenum<tptr->ntrain ; casenum++) { // Do all samples

      inptr = tptr->data + size * casenum ;  // This case is here
      trial ( inptr ) ;                      // Execute the network
      err = 0.0 ;

      if (output_mode == OUTMOD_CLASSIFICATION) { // If Classification mode
         true_class = (int) inptr[tptr->n_inputs] - 1 ; // class is after inputs
         for (i=0 ; i<n_outputs ; i++) {
            if (true_class == i)
               target = neuron_on ;
            else
               target = neuron_off ;         
            errderiv_r ( n_outputs , i , out , target , &err , delta_out ) ;
            }
         }

      else if (output_mode == OUTMOD_MAPPING) {  // If MAPPING mode
         inptr += tptr->n_inputs ;                // outputs after inputs
         for (i=0 ; i<n_outputs ; i++)
            errderiv_r ( n_outputs , i , out , inptr[i] , &err , delta_out ) ;
         }

      if (! outlin) {
         for (i=0 ; i<n_outputs ; i++)
            delta_out[i] *= actderiv ( out[i] ) ;
         }

/*
   Cumulate output gradient
*/

      gradient = grad_out ;
      if (nhid1)                    // If there is a hidden layer
         act_before = before_out ;  // Point to previous layer
      else                          // But if not
         act_before = tptr->data + casenum * size ; // Prev activity is inputs
      for (i=0 ; i<n_outputs ; i++) {
         delta = delta_out[i] ;
         for (j=0 ; j<n_before ; j++)
            *gradient++ += delta * act_before[j] ;
         *gradient++ += delta ;   // Bias
         }

/*
   Cumulate gradient of second hidden layer
*/
   
      if (nhid2) {
         gradient = grad2 ;
         for (i=0 ; i<nhid2 ; i++) {
            delta = 0.0 ;
            for (j=0 ; j<n_outputs ; j++)
               delta += delta_out[j] * out_coefs[j*nhid2_n+i] ;
            delta *= actderiv ( hid2[i] ) ;
            delta2[i] = delta ;
            for (j=0 ; j<nhid1 ; j++)
               *gradient++ += hid1[j] * delta ;
            *gradient++ += delta ;   // Bias
            }
         }

/*
   Cumulate gradient of first hidden layer
*/
   
      if (nhid1) {
         gradient = grad1 ;
         act_before = tptr->data + casenum * size ;
         for (i=0 ; i<nhid1 ; i++) {
            delta = 0.0 ;
            for (j=0 ; j<n_after ; j++)
               delta += delta_after[j] * w_after[j*nhid1_n+i] ;
            delta *= actderiv ( hid1[i] ) ;
            for (j=0 ; j<n_inputs ; j++)
               *gradient++ += delta * act_before[j] ;
            *gradient++ += delta ;   // Bias activation is always 1
            }
         }

      error += err ;                        // Cumulate across epoch 
      } // for all casenums
   
/*
   Find the mean per presentation.  Also, compensate for n_outputs if that was
   not done implicitly in the error computation.
   Remember that if we multiply the error, we must do the same to the gradient.
*/

   factor = 1.0 / ((double) tptr->ntrain  *  (double) nout_n) ;

   for (i=0 ; i<n ; i++)
      grad[i] *= factor ;

   return factor * error ;
}


/*
--------------------------------------------------------------------------------

   GRADIENT_COMPLEX - Gradient for all complex models

--------------------------------------------------------------------------------
*/

double MLFN::gradient_complex (
   TrainingSet *tptr ,
   double *work1 ,
   double *work2 ,
   double *grad
   )
{
   int i, j, k, size, casenum, true_class, n, nprev ;
   double err, error, *dptr, *act_before, *hid1grad, *outgrad, *gradient ;
   double rdiff, idiff, rsum, isum ;
   double factor, rdelta, idelta, target ;
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

   double neuron_on, neuron_off ;

   if (outlin  &&  (errtype != ERRTYPE_XENT)  &&  (errtype != ERRTYPE_KK)) {
      neuron_on = NEURON_ON ;
      neuron_off = NEURON_OFF ;
      }
   else {
      neuron_on = 0.9 * NEURON_ON ;
      neuron_off = 0.9 * NEURON_OFF ;
      }

/*
   Compute size of each training sample
*/

   if (output_mode == OUTMOD_CLASSIFICATION)
      size = tptr->n_inputs + 1 ;
   else if (output_mode == OUTMOD_MAPPING)
      size = tptr->n_inputs + tptr->n_outputs ;

/*
   Compute length of grad vector and gradient positions in it.
   Also compute positions in work1 where the various partial derivatives
   of the output activation are stored.
*/

   if (nhid1 == 0) {      // No hidden layer
      n = n_outputs * nin_n ;
      outgrad = grad ;
      dar10 = work1 ;
      dar01 = dar10 + n_outputs ;
      if (domain == DOMAIN_COMPLEX) {
         dai10 = dar01 + n_outputs ;
         dai01 = dai10 + n_outputs ;
         }
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + n_outputs * nhid1_n ;
      hid1grad = grad ;
      outgrad = hid1grad + nhid1 * nin_n ;
      dhr10 = work1 ;
      dhr01 = dhr10 + nhid1 ;
      if (domain == DOMAIN_COMPLEX_INPUT)
         hend = dhr01 + nhid1 ;
      else {
         dhi10 = dhr01 + nhid1 ;
         dhi01 = dhi10 + nhid1 ;
         hend = dhi01 + nhid1 ;
         }
      dar10 = hend ;
      delta_r = dar10 + n_outputs ;
      if (domain != DOMAIN_COMPLEX_INPUT) {
         dar01 = delta_r + n_outputs ;
         delta_i = dar01 + n_outputs ;
         if (domain == DOMAIN_COMPLEX) {
            dai10 = delta_i + n_outputs ;
            dai01 = dai10 + n_outputs ;
            }
         }
      }

/*
   This is the main loop which cumulates across the epoch
*/
   error = 0.0 ;           // Will cumulate total error here
   for (i=0 ; i<n ; i++)   // Zero gradient for summing
      grad[i] = 0.0 ;
   for (casenum=0 ; casenum<tptr->ntrain ; casenum++) { // Do all samples

      dptr = tptr->data + size * casenum ;     // Point to this sample
      err = 0.0 ;

/*
   Compute all activations and partial derivatives of activations
*/

      if (nhid1 == 0) {                // No hidden layer
         switch (domain) {
            case DOMAIN_COMPLEX_INPUT:
               for (i=0 ; i<n_outputs ; i++)
                  partial_cr ( dptr , out_coefs+i*nin_n , out+i , n_inputs ,
                               dar10+i , dar01+i , outlin ) ;
               break ;
            case DOMAIN_COMPLEX:
               for (i=0 ; i<n_outputs ; i++)
                  partial_cc ( dptr , out_coefs+i*nin_n , out+2*i , n_inputs ,
                               dar10+i , dar01+i , dai10+i , dai01+i , outlin );
               break ;
            } // Switch on domain
         }

      else if (nhid2 == 0) {           // One hidden layer
         switch (domain) {
            case DOMAIN_COMPLEX_INPUT:
               for (i=0 ; i<nhid1 ; i++)
                  partial_cr ( dptr , hid1_coefs+i*nin_n , hid1+i , n_inputs ,
                               dhr10+i , dhr01+i , 0 ) ;
               for (i=0 ; i<n_outputs ; i++) {
                  activity_rr ( hid1 , out_coefs+i*nhid1_n , out+i , nhid1 ,
                                outlin ) ;
                  if (outlin)
                     dar10[i] = 1.0 ;
                  else 
                     dar10[i] = actderiv ( out[i] ) ;
                  }
               break ;
            case DOMAIN_COMPLEX_HIDDEN:
               for (i=0 ; i<nhid1 ; i++)
                  partial_cc ( dptr , hid1_coefs+i*nin_n , hid1+2*i , n_inputs ,
                               dhr10+i , dhr01+i , dhi10+i , dhi01+i , 0 ) ;
               for (i=0 ; i<n_outputs ; i++)
                  partial_cr ( hid1 , out_coefs+i*nhid1_n , out+i , nhid1 ,
                               dar10+i , dar01+i , outlin ) ;
               break ;
            case DOMAIN_COMPLEX:
               for (i=0 ; i<nhid1 ; i++)
                  partial_cc ( dptr , hid1_coefs+i*nin_n , hid1+2*i , n_inputs ,
                               dhr10+i , dhr01+i , dhi10+i , dhi01+i , 0 ) ;
               for (i=0 ; i<n_outputs ; i++)
                  partial_cc ( hid1 , out_coefs+i*nhid1_n , out+2*i , nhid1 ,
                               dar10+i , dar01+i , dai10+i , dai01+i , outlin );
               break ;
            } // Switch on domain
         }

/*
   Compute output error and its derivative with respect to output activation.
   (Put derivative in work2)
*/

      if (output_mode == OUTMOD_CLASSIFICATION) { // If this is Classification
         true_class = (int) dptr[tptr->n_inputs] - 1 ; // class after inputs
         for (i=0 ; i<n_outputs ; i++) {
            if (true_class == i)
               target = neuron_on ;
            else
               target = neuron_off ;
            errderiv_r ( n_outputs , i , out , target , &err , work2 ) ;
            }
         }

      else if (output_mode == OUTMOD_MAPPING) {  // If this is MAPPING output
         dptr += tptr->n_inputs ;                // outputs stored after inputs
         if (domain == DOMAIN_COMPLEX)
            for (i=0 ; i<n_outputs ; i++)
               errderiv_c ( n_outputs , i , out , dptr[2*i] , dptr[2*i+1] ,
                            &err , work2 ) ;
         else
            for (i=0 ; i<n_outputs ; i++)
               errderiv_r ( n_outputs , i , out , dptr[i] , &err , work2 ) ;
         }

      error += err ;                        // Cumulate presentation into epoch

/*
   Cumulate output gradient.  Prevact is the activity in the layer
   just prior to the output layer.
*/

      if (nhid1 == 0) {        // No hidden layer
         nprev = n_inputs ;
         act_before = tptr->data + size * casenum ;
         }
      else {
         nprev = nhid1 ;
         act_before = hid1 ;
         }

      gradient = outgrad ;

      if ((domain == DOMAIN_COMPLEX_INPUT)  &&  nhid1) {  // Real-to-real
         for (i=0 ; i<n_outputs ; i++) { // Do every output neuron
            rdiff = work2[i] ;          // Target minus attained output
            rdelta = rdiff * dar10[i] ;
            delta_r[i] = rdelta ;       // Save for hidden layer computations
            for (j=0 ; j<nprev ; j++)   // Connection to every previous neuron
               *gradient++ += rdelta * act_before[j] ;
            *gradient++ += rdelta ;  // Bias (act_before=1)
            }
         }

      else {                            // Complex domain
         for (i=0 ; i<n_outputs ; i++) { // Do every output neuron
            if (domain == DOMAIN_COMPLEX) { // Complex range
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
               *gradient++ += rdelta * act_before[2*j] + idelta * act_before[2*j+1] ;
               *gradient++ += -rdelta * act_before[2*j+1] + idelta * act_before[2*j] ;
               }  // Next two lines are bias (act_before = 1 + 0 i)
            *gradient++ += rdelta ;
            *gradient++ += idelta ;
            }
         }

/*
   Cumulate hid1 gradient
*/
   
      if (nhid1) {
         act_before = tptr->data + size * casenum ;
         gradient = hid1grad ;

         for (i=0 ; i<nhid1 ; i++) {    // For every hidden neuron

            // Sum out weight * out delta, then multiply by grad to get delta
            rsum = isum = 0.0 ;
            if (domain == DOMAIN_COMPLEX_INPUT) {  // Complex-to-real
               for (k=0 ; k<n_outputs ; k++)    // Sum for all outputs
                  rsum += delta_r[k] * out_coefs[k*nhid1_n+i] ;
               rdelta = rsum * dhr10[i] ;
               idelta = rsum * dhr01[i] ;
               }
            else {  // Complex-to-complex
               for (k=0 ; k<n_outputs ; k++) {  // Sum for all outputs
                  rsum += delta_r[k] * out_coefs[k*nhid1_n+2*i] +
                          delta_i[k] * out_coefs[k*nhid1_n+2*i+1] ;
                  isum += -delta_r[k] * out_coefs[k*nhid1_n+2*i+1] +
                          delta_i[k] * out_coefs[k*nhid1_n+2*i] ;
                  }
               rdelta = rsum * dhr10[i]  +  isum * dhi10[i] ;
               idelta = rsum * dhr01[i]  +  isum * dhi01[i] ;
               }

            // Gradient is delta times previous layer activity
            for (j=0 ; j<n_inputs ; j++) {  // For every input connection
               *gradient++ += rdelta * act_before[2*j] + idelta * act_before[2*j+1];
               *gradient++ += -rdelta * act_before[2*j+1] + idelta* act_before[2*j];
               }  // Next two lines are bias (act_before = 1 + 0 i)
            *gradient++ += rdelta ;
            *gradient++ += idelta ;
            } // For every input connection
         } // If hidden layer

      } // for all casenums
   
/*
   Find the mean per presentation.  Also, compensate for n_outputs if that was
   not done implicitly in the error computation.
   Remember that if we multiply the error, we must do the same to the gradient.
*/

   factor = 1.0 / ((double) tptr->ntrain  *  (double) nout_n) ;

   for (i=0 ; i<n ; i++)
      grad[i] *= factor ;

   return factor * error ;
}

/*
--------------------------------------------------------------------------------

   These two routines compute the (negative) derivative of the network error
   with respect to the output activation.  There is a real and complex version.

--------------------------------------------------------------------------------
*/

void MLFN::errderiv_r (
   int      ,        // Number of output neurons
   int iout ,        // Of which we are doing this one (org 0)
   double *outs ,    // All output activations
   double target ,   // Target for this one
   double *err ,     // Input of error so far, output cumulated per this
   double *deriv     // Output: deriv[iout] = deriv of err wrt outs[iout]
   )
{
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
      }
}

void MLFN::errderiv_c (
   int ,             // Number of output neurons
   int iout ,        // Of which we are doing this one (org 0)
   double *outs ,    // All output activations (real, imaginary, ...)
   double target_r , // Target for this one
   double target_i , // And imaginary part
   double *err ,     // Input of error so far, output cumulated per this
   double *deriv     // Output: deriv[iout] = deriv of err wrt outs[iout]
   )
{
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
         }

}
