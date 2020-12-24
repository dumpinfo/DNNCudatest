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
/*  REGRESS - Use regression to compute LayerNet output weights               */
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

double LayerNet::regress (
   TrainingSet *tptr ,       // Training set used for regression input
   SingularValueDecomp *sptr // Work areas and needed functions
   )

{
   int i, out, tset, size, nvars, is_complex ;
   double *aptr, *bptr, *dptr, err, diff, cdiff[2] ;


/*
   The variable is_complex is 0 if both input and output are real,
   one if input complex and output real, two if both complex.
*/

   if (model == NETMOD_REAL)
      is_complex = 0 ;
   else if (model == NETMOD_COMPLEX_INPUT)
      is_complex = nhid1  ?  0 : 1 ;
   else if (model == NETMOD_COMPLEX_HIDDEN)
      is_complex = 1 ;
   else 
      is_complex = 2 ;

/*
   Compute the size of each training sample in tptr->data and the number
   of independent variables (columns of matrix)
*/

   if (outmod == OUTMOD_CLASSIFY)
      size = tptr->nin + 1 ;
   else if (outmod == OUTMOD_AUTO)
      size = tptr->nin ;
   else if (outmod == OUTMOD_GENERAL)
      size = tptr->nin + tptr->nout ;

   if (nhid1 == 0)         // No hidden layer
      nvars = nin_n ;
   else if (nhid2 == 0)    // One hidden layer
      nvars = nhid1_n ;
   else                    // Two hidden layers
      nvars = nhid2_n ;

/*
   Pass through training set, building matrix of activations prior to
   the output layer, then find its singular value decomposition.
   We keep a copy of it so we can compute the error later.
   If we are mapping real-to-real, the rows are straight copies of activations.
   Otherwise, recall that (a+bi) * (w+vi) = (aw-bv) + (av+bw) i.
   Thus, if mapping complex-to-real, we switch the sign of every other input
   to take care of the real part of the product, and we ignore the imaginary.
   If complex-to-complex, we must also take care of the imaginary part by
   adding another equation.
*/

   aptr = sptr->a ;                     // Will build matrix here

   for (tset=0 ; tset<tptr->ntrain ; tset++) { // Do all training samples

      dptr = tptr->data + size * tset ; // Point to this sample

      if (nhid1 == 0) {                 // No hidden layer, so matrix is inputs
         if (is_complex == 0) {
            for (i=0 ; i<tptr->nin ; i++)
               *aptr++ = *dptr++ ;
            *aptr++ = 1.0 ;             // Bias term is last column of matrix
            }
         else {
            for (i=0 ; i<tptr->nin ; i+=2) {  // Predicting real part
               *aptr++ = *dptr++ ;
               *aptr++ = -*dptr++ ;
               }
            *aptr++ = 1.0 ;
            *aptr++ = 0.0 ;
            if (is_complex == 2) {      // Must also predict imaginary part
               dptr -= tptr->nin ;
               for (i=0 ; i<tptr->nin ; i+=2) {
                  *(aptr+1) = *dptr++ ;
                  *aptr = *dptr++ ;
                  aptr += 2 ;
                  }
               *aptr++ = 0.0 ;
               *aptr++ = 1.0 ;
               }
            }
         }

      else if (nhid2 == 0) {            // One hidden layer
         switch (model) {
            case NETMOD_REAL:
               for (i=0 ; i<nhid1 ; i++) { // so matrix is hidden1 activations
                  activity_rr ( dptr , hid1_coefs+i*nin_n , aptr , nin , 0 ) ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;             // Bias term is last column of matrix
               break ;
            case NETMOD_COMPLEX_INPUT:
               for (i=0 ; i<nhid1 ; i++) {
                  activity_cr ( dptr , hid1_coefs+i*nin_n , aptr , nin , 0 ) ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;
               break ;
            case NETMOD_COMPLEX_HIDDEN:
               for (i=0 ; i<nhid1 ; i++) {
                  activity_cc ( dptr , hid1_coefs+i*nin_n , aptr , nin , 0 ) ;
                  ++aptr ;
                  *aptr = -*aptr ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;
               *aptr++ = 0.0 ;
               break ;
            case NETMOD_COMPLEX:
               for (i=0 ; i<nhid1 ; i++) {
                  activity_cc ( dptr , hid1_coefs+i*nin_n , aptr , nin , 0 ) ;
                  *(aptr+nvars) = *(aptr+1) ; // Next equation which predicts
                  *(aptr+nvars+1) = *aptr ;   // the imaginary part
                  ++aptr ;                    // This is the real part
                  *aptr = -*aptr ;            // of the prediction
                  ++aptr ;
                  }
               *(aptr+nvars) = 0.0 ;          // Bias terms for both equations
               *aptr++ = 1.0 ;
               *(aptr+nvars) = 1.0 ;
               *aptr++ = 0.0 ;
               aptr += nvars ;                // Bypass imaginary equation
               break ;
            } // Switch on model
         }

      else {                            // Two hidden layers
         switch (model) {
            case NETMOD_REAL:
               for (i=0 ; i<nhid1 ; i++)
                  activity_rr ( dptr , hid1_coefs+i*nin_n , hid1+i , nin , 0 ) ;
               for (i=0 ; i<nhid2 ; i++) {
                  activity_rr ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;          // Bias term is last column of matrix
               break ;
            case NETMOD_COMPLEX_INPUT:
               for (i=0 ; i<nhid1 ; i++)
                  activity_cr ( dptr , hid1_coefs+i*nin_n , hid1+i , nin , 0 ) ;
               for (i=0 ; i<nhid2 ; i++) {
                  activity_rr ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;          // Bias term is last column of matrix
               break ;
            case NETMOD_COMPLEX_HIDDEN:
               for (i=0 ; i<nhid1 ; i++)
                  activity_cc ( dptr , hid1_coefs+i*nin_n , hid1+2*i , nin , 0);
               for (i=0 ; i<nhid2 ; i++) {
                  activity_cc ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  ++aptr ;
                  *aptr = -*aptr ;
                  ++aptr ;
                  }
               *aptr++ = 1.0 ;
               *aptr++ = 0.0 ;
               break ;
            case NETMOD_COMPLEX:
               for (i=0 ; i<nhid1 ; i++)
                  activity_cc ( dptr , hid1_coefs+i*nin_n , hid1+2*i , nin , 0);
               for (i=0 ; i<nhid2 ; i++) {
                  activity_cc ( hid1 , hid2_coefs+i*nhid1_n , aptr , nhid1 , 0);
                  *(aptr+nvars) = *(aptr+1) ; // Next equation which predicts
                  *(aptr+nvars+1) = *aptr ;   // the imaginary part
                  ++aptr ;                    // This is the real part
                  *aptr = -*aptr ;            // of the prediction
                  ++aptr ;
                  }
               *(aptr+nvars) = 0.0 ;          // Bias terms for both equations
               *aptr++ = 1.0 ;
               *(aptr+nvars) = 1.0 ;
               *aptr++ = 0.0 ;
               aptr += nvars ;                // Bypass imaginary equation
               break ;
            } // Switch on model
         } // Two layer net
      } // For each training sample


/*
   Do the singular value decomposition.
   Then solve for weights for each output neuron.
   After each output weight vector is computed (using backsub),
   compute the activation of that output neuron, compare it to
   its desired value in the training set, and cumulate the error.
*/

   sptr->svdcmp () ;

   err = 0.0 ;

   for (out=0 ; out<nout_n ; out++) {  // For each output neuron

      if (out % 2  &&  (model == NETMOD_COMPLEX)) // We did the imaginary part
         continue ;                               // when we did the real part

      bptr = sptr->b ;               // Backsub routine wants RHS here

/*
   This pass through the training set computes the desired net inputs to
   the output layer by applying the inverse activation function to the
   target outputs.
*/

      for (tset=0 ; tset<tptr->ntrain ; tset++) {

         dptr = tptr->data + size * tset ;    // Training sample starts here

         if (outmod == OUTMOD_AUTO) { // If AUTOASSOCIATIVE output=input
            if (model == NETMOD_COMPLEX) {
               if (outlin) {
                  bptr[0] = dptr[out] ;
                  bptr[1] = dptr[out+1] ;
                  }
               else 
                  inverse_act_cc ( dptr+out , bptr ) ;
               bptr += 2 ;
               }
            else {
               if (outlin)
                  *bptr++ = dptr[out] ;
               else 
                  *bptr++ = inverse_act ( dptr[out] ) ;
               }
            }

         else if (outmod == OUTMOD_CLASSIFY) { // If this is Classification
            if ((int) dptr[tptr->nin] == out+1) { // class ident past inputs
               if (outlin)
                  *bptr++ = NEURON_ON ;
               else 
                  *bptr++ = inverse_act ( NEURON_ON ) ;
               }
            else {
               if (outlin)
                  *bptr++ = NEURON_OFF ;
               else 
                  *bptr++ = inverse_act ( NEURON_OFF ) ;
               }
            }

         else if (outmod == OUTMOD_GENERAL) { // If this is GENERAL output
            if (model == NETMOD_COMPLEX) {
               if (outlin) {
                  bptr[0] = dptr[tptr->nin+out] ;
                  bptr[1] = dptr[tptr->nin+out+1] ;
                  }
               else 
                  inverse_act_cc ( dptr+tptr->nin+out , bptr ) ;
               bptr += 2 ;
               }
            else {
               if (outlin)
                  *bptr++ = dptr[tptr->nin+out] ;
               else 
                  *bptr++ = inverse_act ( dptr[tptr->nin+out] ) ;
               }
            }
         } // For all training samples

/*
   Find the output weights by back-substitution
*/

      if (model == NETMOD_COMPLEX)
         bptr = out_coefs + out * nvars / 2 ; // Out goes to 2*nout if complex
      else 
         bptr = out_coefs + out * nvars ;

      sptr->backsub ( 1.e-8 , bptr ) ;

/*
   The weights for output neuron 'out' are now in place in out_coefs and are
   pointed to by bptr.  Pass through the training set, using the activations
   of the layer just before the output layer, still in sptr->a, to compute
   the activation of the output neuron.  Compare this attained activation to
   the desired in the training sample, and cumulate the mean square error.
   Note that we use nvars-1 in the call to 'activity' because the bias term
   is taken care of in that subroutine.
   We must also reverse the signs of the imaginary parts of the activations
   because we reversed them before storing.  If this is a full complex model,
   every other row of the activation matrix is for the imaginary part of the
   prediction, and can be ignored because it is redundant.
   If the error type is anything but MSE we cannot do this.  Just skip it
   and call trial_error.
*/

      if (errtype != ERRTYPE_MSE)
         continue ;

      for (tset=0 ; tset<tptr->ntrain ; tset++) {// Cumulate err of this output

         dptr = tptr->data + size * tset ;    // Training sample starts here

         if (is_complex == 2)                 // Point to inputs to output layer
            aptr = sptr->a + 2 * tset * nvars ; // Imaginary row is redundant
         else 
            aptr = sptr->a + tset * nvars ;

         if (is_complex  &&  ! out) {         // Get original activations
            for (i=0 ; i<nvars ; i++) {       // by flipping imaginary part
               if (i % 2)
                  aptr[i] = -aptr[i] ;
               }
            }

         if (is_complex == 0)
            activity_rr ( aptr , bptr , &diff , nvars-1 , outlin ) ;
         else if (is_complex == 1)
            activity_cr ( aptr , bptr , &diff , (nvars-2)/2 , outlin ) ;
         else
            activity_cc ( aptr , bptr , cdiff , (nvars-2)/2 , outlin ) ;

         if (outmod == OUTMOD_AUTO) { // If AUTOASSOCIATIVE, desired out is in
            if (is_complex == 2) {
               cdiff[0] -= dptr[out] ;    // Real part of output
               cdiff[1] -= dptr[out+1] ;  // And imaginary
               err += cdiff[0] * cdiff[0]  +  cdiff[1] * cdiff[1] ;
               }
            else {
               diff -= dptr[out] ;
               err += diff * diff ;
               }
            }

         else if (outmod == OUTMOD_CLASSIFY) {// If this is Classification
            if ((int) dptr[tptr->nin] == out+1)  // class identifier past inputs
               diff -= NEURON_ON ;
            else 
               diff -= NEURON_OFF ;
            err += diff * diff ;
            }

         else if (outmod == OUTMOD_GENERAL) {    // If GENERAL output past input
            if (is_complex == 2) {
               cdiff[0] -= dptr[tptr->nin+out] ;   // Real part of output
               cdiff[1] -= dptr[tptr->nin+out+1] ; // And imaginary
               err += cdiff[0] * cdiff[0]  +  cdiff[1] * cdiff[1] ;
               }
            else {
               diff -= dptr[tptr->nin+out] ;
               err += diff * diff ;
               }
            }
         } // For all tsets
      } // For each output

   if (errtype != ERRTYPE_MSE)
      neterr = trial_error ( tptr ) ; // Did we skip above error computation?
   else 
      neterr = 25. * err / ((double) tptr->ntrain * (double) nout) ;
   return neterr ;
}

