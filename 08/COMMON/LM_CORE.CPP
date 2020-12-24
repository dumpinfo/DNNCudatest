/******************************************************************************/
/*                                                                            */
/*  LM_CORE - Called by LEV_MARQ's criter to compute error, hess, and grad    */
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

double MLFN::lm_core (
   TrainingSet *tptr ,
   double *work1 ,
   double *work2 ,
   double *hessian ,
   double *gradient
   )
{
   if (domain == DOMAIN_REAL)
      return lm_core_real ( tptr , work1 , work2 , hessian , gradient ) ;
   else 
      return lm_core_complex ( tptr , work1 , work2 , hessian , gradient ) ;
}

/*
--------------------------------------------------------------------------------

   LM_CORE_REAL - For pure real model

   This passes through the entire training set, cumulating hessian and gradient.
   For each case, 'process_real' is called to do the actual work.

--------------------------------------------------------------------------------
*/

double MLFN::lm_core_real (
   TrainingSet *tptr ,
   double *hid2delta ,
   double *grad ,
   double *hessian ,
   double *gradient
   )
{
   int i, j, size, tset, tclass ;
   double err, error, *dptr, targ ;
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
   Compute length of grad vector (number of parameters).
*/

   for (i=0 ; i<ntot ; i++) { // Zero hessian and gradient for summing
      gradient[i] = 0.0 ;
      for (j=0 ; j<=i ; j++)  // Symmetric, so only cumulate half
         hessian[i*ntot+j] = 0.0 ;
      }

   error = 0.0 ;  // Will cumulate total error here

   for (tset=0 ; tset<tptr->ntrain ; tset++) { // Do all samples

      dptr = tptr->data + size * tset ;     // Point to this sample
      trial ( dptr ) ;                      // Evaluate network for it
      err = 0.0 ;                           // Cumulates for this presentation

      if (output_mode == OUTMOD_CLASSIFICATION) { // If this is Classification
         tclass = (int) dptr[tptr->n_inputs] - 1 ; // class is stored after inputs
         for (i=0 ; i<n_outputs ; i++) {         // Recall that train added a
            if (tclass == i)                // fraction so that the above
               targ = neuron_on ;           // truncation to get tclass is
            else                            // always safe in any radix
               targ = neuron_off ;         
            process_real ( dptr , i , targ , &err ,
                           hessian , gradient , hid2delta , grad ) ;
            }
         }

      else if (output_mode == OUTMOD_MAPPING) {  // Outputs stored after inputs
         for (i=0 ; i<n_outputs ; i++)
            process_real ( dptr , i , (dptr+tptr->n_inputs)[i] , &err ,
                           hessian , gradient , hid2delta , grad ) ;
         }

      error += err ;
      } // for all tsets

/*
   Fill in the uncomputed half of hessian via symmetry.
   Find the mean per presentation.  Also, compensate for n_outputs.
*/

   for (i=1 ; i<ntot ; i++) {
      for (j=0 ; j<i ; j++)
         hessian[j*ntot+i] = hessian[i*ntot+j] ;
      }

   return error  / ((double) tptr->ntrain  *  (double) nout_n) ;
}

/*
--------------------------------------------------------------------------------

   process_real - Called by lm_core_real to handle each case.

--------------------------------------------------------------------------------
*/

void MLFN::process_real (
   double *input ,
   int idep ,
   double target ,
   double *err ,
   double *hessian ,
   double *gradient ,
   double *hid2delta ,
   double *grad
   )
{
   int i, j, n, nprev ;
   double delta, *hid1grad, *hid2grad, *outgrad, outdelta ;
   double *outprev, *prevact, *gradptr, diff, *aptr ;

/*
   Compute the length of the gradient vector and various positions in it.
   Also point to the layer just before the output.
   For nprev we use the _w version because they refer to actual
   neuron activations.  Bias is handled separately.
*/

   if (nhid1 == 0) {      // No hidden layer
      n = n_outputs * nin_n ;
      outgrad = grad ;
      nprev = nin_w ;
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + n_outputs * nhid1_n ;
      hid1grad = grad ;
      outgrad = grad + nhid1 * nin_n ;
      outprev = hid1 ;
      nprev = nhid1_w ;
      }
   else {                 // Two hidden layers
      n = nhid1 * nin_n + nhid2 * nhid1_n + n_outputs * nhid2_n ;
      hid1grad = grad ;
      hid2grad = grad + nhid1 * nin_n ;
      outgrad = hid2grad + nhid2 * nhid1_n ;
      outprev = hid2 ;
      nprev = nhid2_w ;
      }

/*
   Save the derivative of each output activation with respect to net input
*/

   if (outlin)
      outdelta = 1.0 ;
   else
      outdelta = actderiv ( out[idep] ) ;

/*
   Compute output gradient.  Prevact is the activity in the layer
   just prior to the output layer.
*/

   if (nhid1 == 0)         // If no hidden layer
      prevact = input ;    // Then direct connection to inputs
   else
      prevact = outprev ;  // Point to previous layer

   gradptr = outgrad ;
   for (i=0 ; i<n_outputs ; i++) {
      if (i == idep) {
         for (j=0 ; j<nprev ; j++)
            *gradptr++ = outdelta * prevact[j] ;
         *gradptr++ = outdelta ;   // Bias activation is always 1
         }
      else {
         for (j=0 ; j<nprev ; j++)
            *gradptr++ = 0.0 ;
         *gradptr++ = 0.0 ;
         }
      }

/*
   Cumulate gradient of second hidden layer
*/
   
   if (nhid2) {
      gradptr = hid2grad ;
      for (i=0 ; i<nhid2 ; i++) {
         delta = outdelta * out_coefs[idep*nhid2_n+i] ;
         delta *= actderiv ( hid2[i] ) ;
         hid2delta[i] = delta ;
         for (j=0 ; j<nhid1 ; j++)
            *gradptr++ = delta * hid1[j] ;
         *gradptr++ = delta ;   // Bias activation is always 1
         }
      }

/*
   Cumulate the gradient of the first hidden layer.
   If there is  hid2 layer, we must use the generalized chain rule
   to sum across those neurons.  If not, then we only do the 'idep'
   output neuron.
*/
   
   if (nhid1) {
      gradptr = hid1grad ;
      prevact = input ;
      for (i=0 ; i<nhid1 ; i++) {
         if (nhid2) {
            delta = 0.0 ;
            for (j=0 ; j<nhid2 ; j++)
               delta += hid2delta[j] * hid2_coefs[j*nhid1_n+i] ;
            }
         else 
            delta = outdelta * out_coefs[idep*nhid1_n+i] ;
         delta *= actderiv ( hid1[i] ) ;
         for (j=0 ; j<n_inputs ; j++)
            *gradptr++ = delta * prevact[j] ;
         *gradptr++ = delta ;   // Bias activation is always 1
         }
      }

   diff = target - out[idep] ; // Target minus attained output
   *err += diff * diff ;

   for (i=0 ; i<n ; i++) {
      gradient[i] += diff * grad[i] ;
      aptr = hessian + i*n ;
      for (j=0 ; j<=i ; j++)
         aptr[j] += grad[i] * grad[j] ;
      }
}

/*
--------------------------------------------------------------------------------

   lm_core_complex - Handle all complex models

--------------------------------------------------------------------------------
*/

double MLFN::lm_core_complex (
   TrainingSet *tptr ,
   double *work1 ,
   double *grad ,
   double *hessian ,
   double *gradient
   )
{
   int i, j, size, tset, tclass, n ;
   double err, error, *dptr, targ ;
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
*/

   if (nhid1 == 0) {      // No hidden layer
      n = n_outputs * nin_n ;
      dar10 = work1 ;
      dar01 = dar10 + n_outputs ;
      if (domain == DOMAIN_COMPLEX) {
         dai10 = dar01 + n_outputs ;
         dai01 = dai10 + n_outputs ;
         }
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + n_outputs * nhid1_n ;
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

   for (i=0 ; i<n ; i++) {    // Zero hessian and gradient for summing
      gradient[i] = 0.0 ;
      for (j=0 ; j<=i ; j++)  // Symmetric, so only cumulate half
         hessian[i*n+j] = 0.0 ;
      }

   error = 0.0 ;  // Will cumulate total error here

/*
   This is the main loop that cumulates across the epoch
*/

   for (tset=0 ; tset<tptr->ntrain ; tset++) { // Do all samples

      dptr = tptr->data + size * tset ;     // Point to this sample
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
   Process this case
*/

      if (output_mode == OUTMOD_CLASSIFICATION) { // If this is Classification
         tclass = (int) dptr[tptr->n_inputs] - 1 ; // class is stored after inputs
         for (i=0 ; i<n_outputs ; i++) {         // Recall that train added a
            if (tclass == i)                // fraction so that the above
               targ = neuron_on          ;  // truncation to get tclass is
            else                            // always safe in any radix
               targ = neuron_off ;
            process_cr ( dptr , i , targ , &err ,
                         hessian , gradient , work1 , grad ) ;
            }
         }

      else if (output_mode == OUTMOD_MAPPING) {  // If this is MAPPING output
         dptr += tptr->n_inputs ;                // outputs stored after inputs
         if (domain == DOMAIN_COMPLEX)
            for (i=0 ; i<n_outputs ; i++) {
               process_cc ( dptr-tptr->n_inputs , i , dptr[2*i] , 0 , &err ,
                            hessian , gradient , work1 , grad ) ;
               process_cc ( dptr-tptr->n_inputs , i , dptr[2*i+1] , 1 , &err ,
                            hessian , gradient , work1 , grad ) ;
               }
         else
            for (i=0 ; i<n_outputs ; i++)
               process_cr ( dptr-tptr->n_inputs , i , dptr[i] , &err ,
                            hessian , gradient , work1 , grad ) ;
         }

      error += err ;                        // Cumulate presentation into epoch
      } // for all tsets
   
/*
   Fill in the uncomputed half of hessian via symmetry.
   Find the mean per presentation.  Also, compensate for n_outputs.
   Finally, scale in such a way that the result is 0-100,
   for user's convenience.
*/

   for (i=1 ; i<n ; i++) {
      for (j=0 ; j<i ; j++)
         hessian[j*n+i] = hessian[i*n+j] ;
      }

   return error  / ((double) tptr->ntrain  *  (double) nout_n) ;
}

/*
--------------------------------------------------------------------------------

   Process a case in model with real output

--------------------------------------------------------------------------------
*/

void MLFN::process_cr (
   double *input ,
   int idep ,
   double target ,
   double *err ,
   double *hessian ,
   double *gradient ,
   double *work1 ,
   double *grad
   )
{
   int i, j, n, nprev ;
   double *prevact, *hid1grad, *outgrad, *gradptr, *aptr ;
   double diff, rsum, isum ;
   double rdelta, idelta ;
   double *dar10 ;   // Partial of real attained output wrt real net
   double *dar01 ;   // Partial of real attained output wrt imaginary net
   double *dhr10 ;   // Partial of real hidden wrt real net
   double *dhi10 ;   // Partial of imaginary hidden wrt real net
   double *dhr01 ;   // Partial of real hidden wrt imaginary net
   double *dhi01 ;   // Partial of imaginary hidden wrt imaginary net
   double *hend  ;   // Points to next byte after dh's
   double *delta_r ; // Real part of delta

   if (nhid1 == 0) {      // No hidden layer
      n = n_outputs * nin_n ;
      outgrad = grad ;
      dar10 = work1 ;
      dar01 = dar10 + n_outputs ;
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
      delta_r = dar10 + n_outputs ;  // Not needed
      if (domain != DOMAIN_COMPLEX_INPUT)
         dar01 = delta_r + n_outputs ;
      }

/*
   Compute output gradient.  Prevact is the activity in the layer
   just prior to the output layer.
*/

   if (nhid1 == 0) {        // No hidden layer
      nprev = n_inputs ;
      prevact = input ;
      }
   else {
      nprev = nhid1 ;
      prevact = hid1 ;
      }

   gradptr = outgrad ;

   if ((domain == DOMAIN_COMPLEX_INPUT)  &&  nhid1) {  // Real-to-real
      for (i=0 ; i<n_outputs ; i++) {     // Do every output neuron
         if (i == idep) {
            rdelta = dar10[i] ;
            for (j=0 ; j<nprev ; j++)   // Connection to every previous neuron
               *gradptr++ = rdelta * prevact[j] ;
            *gradptr++ = rdelta ;       // Bias (prevact=1)
            }
         else {
            for (j=0 ; j<nprev ; j++)
               *gradptr++ = 0.0 ;
            *gradptr++ = 0.0 ;
            }
         }
      }

   else {                            // Complex domain
      for (i=0 ; i<n_outputs ; i++) {     // Do every output neuron
         if (i == idep) {
            rdelta = dar10[i] ;
            idelta = dar01[i] ;
            for (j=0 ; j<nprev ; j++) { // Connect to every previous neuron
               *gradptr++ = rdelta * prevact[2*j] + idelta * prevact[2*j+1] ;
               *gradptr++ = -rdelta * prevact[2*j+1] + idelta * prevact[2*j] ;
               }  // Next two lines are bias (prevact = 1 + 0 i)
            *gradptr++ = rdelta ;
            *gradptr++ = idelta ;
            }
         else {
            for (j=0 ; j<nprev ; j++) { // Connection to every previous neuron
               *gradptr++ = 0.0 ;
               *gradptr++ = 0.0 ;
               }  // Next two lines are bias (prevact = 1 + 0 i)
            *gradptr++ = 0.0 ;
            *gradptr++ = 0.0 ;
            }
         }
      }

/*
   Cumulate hid1 gradient (if it exists)
*/
   
   if (nhid1) {
      prevact = input ;
      gradptr = hid1grad ;

      for (i=0 ; i<nhid1 ; i++) {    // For every hidden neuron

         if (domain == DOMAIN_COMPLEX_INPUT) {  // Complex-to-real
            rsum = dar10[idep] * out_coefs[idep*nhid1_n+i] ;
            rdelta = rsum * dhr10[i] ;
            idelta = rsum * dhr01[i] ;
            }
         else {  // Complex-to-complex
            rsum = dar10[idep] * out_coefs[idep*nhid1_n+2*i] +
                   dar01[idep] * out_coefs[idep*nhid1_n+2*i+1] ;
            isum = -dar10[idep] * out_coefs[idep*nhid1_n+2*i+1] +
                   dar01[idep] * out_coefs[idep*nhid1_n+2*i] ;
            rdelta = rsum * dhr10[i]  +  isum * dhi10[i] ;
            idelta = rsum * dhr01[i]  +  isum * dhi01[i] ;
            }

         // Gradient is delta times previous layer activity
         for (j=0 ; j<n_inputs ; j++) {  // For every input connection
            *gradptr++ = rdelta * prevact[2*j] + idelta * prevact[2*j+1];
            *gradptr++ = -rdelta * prevact[2*j+1] + idelta* prevact[2*j];
            }  // Next two lines are bias (prevact = 1 + 0 i)
         *gradptr++ = rdelta ;
         *gradptr++ = idelta ;
         } // For every input connection
      } // If hidden layer

   diff = target - out[idep] ; // Target minus attained output
   *err += diff * diff ;

   for (i=0 ; i<n ; i++) {
      gradient[i] += diff * grad[i] ;
      aptr = hessian + i*n ;
      for (j=0 ; j<=i ; j++)
         aptr[j] += grad[i] * grad[j] ;
      }
}

/*
--------------------------------------------------------------------------------

   Process a case in model with complex output

--------------------------------------------------------------------------------
*/

void MLFN::process_cc (
   double *input ,
   int idep ,
   double target ,
   int imag_part ,
   double *err ,
   double *hessian ,
   double *gradient ,
   double *work1 ,
   double *grad
   )
{
   int i, j, n, nprev ;
   double *prevact, *hid1grad, *outgrad, *gradptr, *aptr ;
   double diff, rsum, isum ;
   double rdelta, idelta ;
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

   if (nhid1 == 0) {      // No hidden layer
      n = n_outputs * nin_n ;
      outgrad = grad ;
      dar10 = work1 ;
      dar01 = dar10 + n_outputs ;
      dai10 = dar01 + n_outputs ;
      dai01 = dai10 + n_outputs ;
      }
   else if (nhid2 == 0) { // One hidden layer
      n = nhid1 * nin_n + n_outputs * nhid1_n ;
      hid1grad = grad ;
      outgrad = hid1grad + nhid1 * nin_n ;
      dhr10 = work1 ;
      dhr01 = dhr10 + nhid1 ;
      dhi10 = dhr01 + nhid1 ;
      dhi01 = dhi10 + nhid1 ;
      hend = dhi01 + nhid1 ;
      dar10 = hend ;
      delta_r = dar10 + n_outputs ;
      dar01 = delta_r + n_outputs ;
      delta_i = dar01 + n_outputs ;
      dai10 = delta_i + n_outputs ;
      dai01 = dai10 + n_outputs ;
      }

/*
   Compute output gradient.  Prevact is the activity in the layer
   just prior to the output layer.
*/

   if (nhid1 == 0) {        // No hidden layer
      nprev = n_inputs ;
      prevact = input ;
      }
   else {
      nprev = nhid1 ;
      prevact = hid1 ;
      }

   gradptr = outgrad ;

   for (i=0 ; i<n_outputs ; i++) {     // Do every output neuron
      if (i == idep) {            // Other outputs have zero derivs
         if (imag_part) {         // Imaginary part of output?
            rdelta = dai10[i] ;
            idelta = dai01[i] ;
            }
         else {                   // Real part of output
            rdelta = dar10[i] ;
            idelta = dar01[i] ;
            }
         for (j=0 ; j<nprev ; j++) { // Connect to every previous neuron
            *gradptr++ = rdelta * prevact[2*j] + idelta * prevact[2*j+1] ;
            *gradptr++ = -rdelta * prevact[2*j+1] + idelta * prevact[2*j] ;
            }  // Next two lines are bias (prevact = 1 + 0 i)
         *gradptr++ = rdelta ;
         *gradptr++ = idelta ;
         }
      else {
         for (j=0 ; j<nprev ; j++) { // Connection to every previous neuron
            *gradptr++ = 0.0 ;
            *gradptr++ = 0.0 ;
            }  // Next two lines are bias (prevact = 1 + 0 i)
         *gradptr++ = 0.0 ;
         *gradptr++ = 0.0 ;
         }
      if (nhid1) {                // Save for hidden layer grad
         delta_r[i] = rdelta ;
         delta_i[i] = idelta ;
         }
      }

/*
   Cumulate hid1 gradient (if it exists)
*/
   
   if (nhid1) {
      prevact = input ;
      gradptr = hid1grad ;

      for (i=0 ; i<nhid1 ; i++) {    // For every hidden neuron

         rsum = delta_r[idep] * out_coefs[idep*nhid1_n+2*i] +
                delta_i[idep] * out_coefs[idep*nhid1_n+2*i+1] ;
         isum = -delta_r[idep] * out_coefs[idep*nhid1_n+2*i+1] +
                delta_i[idep] * out_coefs[idep*nhid1_n+2*i] ;
         rdelta = rsum * dhr10[i]  +  isum * dhi10[i] ;
         idelta = rsum * dhr01[i]  +  isum * dhi01[i] ;

         // Gradient is delta times previous layer activity
         for (j=0 ; j<n_inputs ; j++) {  // For every input connection
            *gradptr++ = rdelta * prevact[2*j] + idelta * prevact[2*j+1];
            *gradptr++ = -rdelta * prevact[2*j+1] + idelta* prevact[2*j];
            }  // Next two lines are bias (prevact = 1 + 0 i)
         *gradptr++ = rdelta ;
         *gradptr++ = idelta ;
         } // For every hidden neuron
      } // If hidden layer

   if (imag_part)                     // Imaginary part of output?
      diff = target - out[2*idep+1] ;
   else                               // Real part of output
      diff = target - out[2*idep] ;   // Target minus attained output
   *err += diff * diff ;

   for (i=0 ; i<n ; i++) {
      gradient[i] += diff * grad[i] ;
      aptr = hessian + i*n ;
      for (j=0 ; j<=i ; j++)
         aptr[j] += grad[i] * grad[j] ;
      }
}

