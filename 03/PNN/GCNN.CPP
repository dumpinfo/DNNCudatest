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
/*  GCNN - All principal routines for GCNN processing                         */
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

extern double univar ( double z , GCNNcoefs *c , int mom , int ew ) ;
extern void hermite ( const double x ) ;
static double h2, h3, h4, h5, h6, h7, h8, h9 ;

/*
--------------------------------------------------------------------------------

   Constructor

--------------------------------------------------------------------------------
*/

GCNN::GCNN (
   int max_moment ,
   int edgeworth ,
   int out_model ,
   int n_inputs ,
   int n_outputs
   )
   : PNNet ( 0 , out_model , n_inputs , n_outputs )
{
   if (! ok)    // Did the parent constructor fail?
      return ;  // If so, nothing to do here

   maxmom = max_moment ;
   ewmod = edgeworth ;

   if (edgeworth)
      model = NETMOD_GCNN_EW ;  // Seems silly in C++, but helps sometimes
   else
      model = NETMOD_GCNN ;

   MEMTEXT ( "GCNN constructor (2)" ) ;
   c = (GCNNcoefs *) MALLOC ( n_inputs * n_outputs * sizeof(GCNNcoefs) ) ;
   priors = (double *) MALLOC ( n_outputs * sizeof(double) ) ;

   if ((c == NULL)  ||  (priors == NULL)) {
      if (c != NULL)
         FREE ( c ) ;
      if (priors != NULL)
         FREE ( priors ) ;
      ok = 0 ;
      }
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

GCNN::~GCNN()
{
   if (! ok)    // If constructor's mallocs failed
      return ;  // there is nothing to free

   MEMTEXT ( "GCNN destructor (2)" ) ;
   FREE ( c ) ;
   FREE ( priors ) ;
}

/*
--------------------------------------------------------------------------------

   These are two unused dummy routines.
   They must be here because it is convenient to make them virtual
   in the PNN base class.

--------------------------------------------------------------------------------
*/

int GCNN::trial_deriv ( double * , int , double * )
{
   return 0 ;
}

int GCNN::privatize_tset ()
{
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   trial - Compute the output for a given input by evaluating the network
           This also returns the subscript of the maximum output.

--------------------------------------------------------------------------------
*/

int GCNN::trial ( double *input )
{
   int i, oclass, bestclass ;
   double z, sum, biggest ;
   GCNNcoefs *cptr ;

   sum = 1.e-40 ;
   for (oclass=0 ; oclass<nout ; oclass++) {
      out[oclass] = priors[oclass] ;
      for (i=0 ; i<nin ; i++) {
         cptr = c + oclass * nin ;
         z = (input[i] - cptr[i].mean) / cptr[i].std ; // Standardize
         out[oclass] *= univar ( z , cptr+i , maxmom , ewmod ) ;
         }
      sum += out[oclass] ;
      }

   biggest = -1.0 ;
   for (oclass=0 ; oclass<nout ; oclass++) {
      out[oclass] /= sum ;
      if (out[oclass] > biggest) {
         biggest = out[oclass] ;
         bestclass = oclass ;
         }
      }

   return bestclass ;
}


/*
--------------------------------------------------------------------------------

   learn - Compute the coefficients

--------------------------------------------------------------------------------
*/

void GCNN::learn ( TrainingSet *tptr , struct LearnParams * )
{
   int i, oclass, tset, n ;
   double diff, prod, sum, *dptr, m3, m4, m5 ;
   GCNNcoefs *cptr ;

/*
   For the pure PNN models, we keep the training set as a fundamental part
   of the trained network.  Therefore we take the easy route and just use
   the priors that are part of the training set.  However, the GCNN does not
   need the training set.  So now we must keep a private copy of the priors.
   Recall that the user has the option of declaring that the class counts
   determine the priors.  This fact is flagged internally by setting the
   priors to -1.  (We trust that the user is consistent in this regard!)
   So now we check for that desire, and compute the priors accordingly.
   Also make sure that they sum to 1.
*/

   memcpy ( priors , tptr->priors , nout * sizeof(double) ) ;

   sum = 0.0 ;
   for (i=0 ; i<nout ; i++) {
      if (priors[i] < 0.0)
         priors[i] = tptr->nper[i] ;
      sum += priors[i] ;
      }

   for (i=0 ; i<nout ; i++)
      priors[i] /= sum ;


/*
   Compute the mean and standard deviation for standardizing the data
*/

   for (oclass=0 ; oclass<nout ; oclass++) {  // Zero means and stds
      for (i=0 ; i<nin ; i++) {               // Might as well do coefs too
         cptr = c + oclass * nin + i ;
         cptr->mean = cptr->std = 0.0 ;
         cptr->c3 = cptr->c4 = cptr->c5 = cptr->c6 = cptr->c7 = cptr->cx = 0.0 ;
         }
      }

   for (tset=0 ; tset<tptr->ntrain ; tset++) {  // Do all training cases
      dptr = tptr->data + tptr->size * tset ;   // Point to this case
      oclass = (int) dptr[nin] - 1 ;            // Class stored after inputs
      cptr = c + oclass * nin ;                 // Coef sets for this class
      for (i=0 ; i<nin ; i++)                   // Each input var separate
         cptr[i].mean += dptr[i] ;              // Cumulate mean
      }

   for (oclass=0 ; oclass<nout ; oclass++) {    // Divide sums to get means
      if ((n = (tptr->nper)[oclass]) == 0)      // Insurance
         continue ;
      for (i=0 ; i<nin ; i++) {
         cptr = c + oclass * nin + i ;
         cptr->mean /= n ;
         }
      }

   for (tset=0 ; tset<tptr->ntrain ; tset++) {  // Do all training cases
      dptr = tptr->data + tptr->size * tset ;   // Point to this case
      oclass = (int) dptr[nin] - 1 ;            // Class stored after inputs
      cptr = c + oclass * nin ;                 // Coef sets for this class
      for (i=0 ; i<nin ; i++) {                 // Each input var separate
         diff = dptr[i] - cptr[i].mean ;        // This point minus mean
         cptr[i].std += diff * diff ;           // Cumulate sum of squares
         }
      }

   for (oclass=0 ; oclass<nout ; oclass++) {    // Divide, sqrt to get std
      if ((n = (tptr->nper)[oclass]) == 0)      // Insurance
         continue ;
      for (i=0 ; i<nin ; i++) {
         cptr = c + oclass * nin + i ;
         cptr->std = sqrt ( cptr->std / n ) ;
         }
      }

/*
   The means and standard deviations have been computed.
   Now compute either the Edgeworth or Gram-Charlier coefs.
*/

   if (ewmod) {  // Edgeworth?

/*
   Compute the moments.  Temporarily store them in the e3-e5 coef areas.
*/

      for (tset=0 ; tset<tptr->ntrain ; tset++) {  // Do all training cases
         dptr = tptr->data + tptr->size * tset ;   // Point to this case
         oclass = (int) dptr[nin] - 1 ;            // Class stored after inputs
         cptr = c + oclass * nin ;                 // Coef sets for this class
         for (i=0 ; i<nin ; i++) {                 // Each input var separate
            diff = (dptr[i] - cptr[i].mean) / cptr[i].std ; // Standardize
            prod = diff * diff * diff ;
            cptr[i].c3 += prod ;
            prod *= diff ;
            cptr[i].c4 += prod ;
            prod *= diff ;
            cptr[i].c5 += prod ;
            }
         }

      for (oclass=0 ; oclass<nout ; oclass++) { // Compute moments, then coefs
         if ((n = (tptr->nper)[oclass]) == 0)   // Insurance
            continue ;
         for (i=0 ; i<nin ; i++) {
            cptr = c + oclass * nin ;
            m3 = cptr[i].c3 / n ;               // Moments
            m4 = cptr[i].c4 / n ;
            m5 = cptr[i].c5 / n ;
            cptr[i].c3 = m3 / 6.0 ;
            cptr[i].c4 = (m4 - 3.0) / 24.0 ;
            cptr[i].c5 = (m5 - 10.0 * m3) / 120.0 ;
            cptr[i].c6 = m3 * m3 / 72.0 ;
            cptr[i].c7 = (m4 - 3.0) * m3 * 35.0 / 5040.0 ;
            cptr[i].cx = m3 * m3 * m3 * 280.0 / 362880.0 ;
            }
         }
      } // Edgeworth

   else {   // Else ordinary GC, not Edgeworth

/*
   Compute the GC coefficients via direct projection on the Hermite polynomials
*/

      for (tset=0 ; tset<tptr->ntrain ; tset++) {  // Do all training cases
         dptr = tptr->data + tptr->size * tset ;   // Point to this case
         oclass = (int) dptr[nin] - 1 ;            // Class stored after inputs
         cptr = c + oclass * nin ;                 // Coef sets for this class
         for (i=0 ; i<nin ; i++) {                 // Each input var separate
            diff = (dptr[i] - cptr[i].mean) / cptr[i].std ; // Standardize
            hermite ( diff ) ;                     // Evaluate Hermite polynomials
            cptr[i].c3 += h3 ;
            cptr[i].c4 += h4 ;
            cptr[i].c5 += h5 ;
            cptr[i].c6 += h6 ;
            cptr[i].c7 += h7 ;
            cptr[i].cx += h8 ;
            }
         }

      for (oclass=0 ; oclass<nout ; oclass++) {     // Divide by n and k!
         if ((n = (tptr->nper)[oclass]) == 0)       // Insurance
            continue ;
         for (i=0 ; i<nin ; i++) {
            cptr = c + oclass * nin ;
            cptr[i].c3 /= (n * 6.0) ;
            cptr[i].c4 /= (n * 24.0) ;
            cptr[i].c5 /= (n * 120.0) ;
            cptr[i].c6 /= (n * 720.0) ;
            cptr[i].c7 /= (n * 5040.0) ;
            cptr[i].cx /= (n * 40320.0) ;
            }
         }
      } // Else ordinary GC

   neterr = trial_error ( tptr , 0 ) ;
   errtype = 1 ;          // Tell other routines net is trained
   return ;
}

/*
--------------------------------------------------------------------------------

   wt_save - Save network to disk (called from WT_SAVE.CPP)
   wt_restore - Restore network from disk (called from WT_SAVE.CPP)

--------------------------------------------------------------------------------
*/

int GCNN::wt_save ( FILE *fp )
{
   fwrite ( c , nin * nout * sizeof(GCNNcoefs) , 1 , fp ) ;
   fwrite ( priors , nout * sizeof(double) , 1 , fp ) ;
   if (ferror ( fp ))
      return 1 ;
   return 0 ;
}

void GCNN::wt_restore ( FILE *fp )
{
   fread ( c , nin * nout * sizeof(GCNNcoefs) , 1 , fp ) ;
   fread ( priors , nout * sizeof(double) , 1 , fp ) ;

   if (ferror ( fp ))
      ok = 0 ;
   return ;
}


/*
--------------------------------------------------------------------------------

   Hermite - Local routine to compute Hermite polynomials

--------------------------------------------------------------------------------
*/

static void hermite ( const double x )
{
   h2 = x * x - 1.0 ;
   h3 = x * h2 - 2.0 * x ;
   h4 = x * h3 - 3.0 * h2 ;
   h5 = x * h4 - 4.0 * h3 ;
   h6 = x * h5 - 5.0 * h4 ;
   h7 = x * h6 - 6.0 * h5 ;
   h8 = x * h7 - 7.0 * h6 ;
   h9 = x * h8 - 8.0 * h7 ;
}

/*
--------------------------------------------------------------------------------

   univar - Local routine to compute univariate GC or EW likelihood

--------------------------------------------------------------------------------
*/

static double univar ( double z , GCNNcoefs *c , int mom , int ew )
{
   double a, b ;

   a = exp ( -0.5 * z * z ) ;   // Normal density (times a constant)
   h2 = z * z - 1.0 ;           // Second-degree Hermite polynomial
   h3 = z * (h2 - 2.0) ;        // And third
   b = 1.0  +  c->c3 * h3 ;     // First extra moment term (3rd moment)

   if (ew) {                    // Edgeworth
      switch (mom) {
         case 4: h4 = z * h3 - 3.0 * h2 ;
                 h5 = z * h4 - 4.0 * h3 ;
                 h6 = z * h5 - 5.0 * h4 ;
                 b += c->c4 * h4 + c->c6 * h6 ;
                 break ;
         case 5: h4 = z * h3 - 3.0 * h2 ;
                 h5 = z * h4 - 4.0 * h3 ;
                 h6 = z * h5 - 5.0 * h4 ;
                 h7 = z * h6 - 6.0 * h5 ;
                 h8 = z * h7 - 7.0 * h6 ;
                 h9 = z * h8 - 8.0 * h7 ;
                 b += c->c4 * h4 + c->c5 * h5 + c->c6 * h6 +
                      c->c7 * h7 + c->cx * h9 ;
                 break ;
         }
      }

   else {  // Else must be ordinary GC
      switch (mom) {
         case 4: h4 = z * h3 - 3.0 * h2 ;
                 b += c->c4 * h4 ;
                 break ;
         case 5: h4 = z * h3 - 3.0 * h2 ;
                 h5 = z * h4 - 4.0 * h3 ;
                 b += c->c4 * h4 + c->c5 * h5 ;
                 break ;
         case 6: h4 = z * h3 - 3.0 * h2 ;
                 h5 = z * h4 - 4.0 * h3 ;
                 h6 = z * h5 - 5.0 * h4 ;
                 b += c->c4 * h4 + c->c5 * h5 + c->c6 * h6 ;
                 break ;
         case 7: h4 = z * h3 - 3.0 * h2 ;
                 h5 = z * h4 - 4.0 * h3 ;
                 h6 = z * h5 - 5.0 * h4 ;
                 h7 = z * h6 - 6.0 * h5 ;
                 b += c->c4 * h4 + c->c5 * h5 + c->c6 * h6 + c->c7 * h7 ;
                 break ;
         case 8: h4 = z * h3 - 3.0 * h2 ;
                 h5 = z * h4 - 4.0 * h3 ;
                 h6 = z * h5 - 5.0 * h4 ;
                 h7 = z * h6 - 6.0 * h5 ;
                 h8 = z * h7 - 7.0 * h6 ;
                 b += c->c4 * h4 + c->c5 * h5 + c->c6 * h6 +
                      c->c7 * h7 + c->cx * h8 ;
                 break ;
         }
      }

   if (b < 1.e-40)
      b = 1.e-40 ;

   return a * b ;
}
