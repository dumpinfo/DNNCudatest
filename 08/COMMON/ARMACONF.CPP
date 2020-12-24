/******************************************************************************/
/*                                                                            */
/*  ARMACONF - Find prediction confidence using an ARMA model                 */
/*                                                                            */
/*  Normally, this returns 0.  It returns -1 if insufficient memory,          */
/*  1 if user pressed ESCape, 2 if too few cases for reliable results.        */
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

#define DEBUG 0

int armaconf (
   int npred ,            // Number of future predictions
   ARMA *arma ,           // Use this ARMA
   int nio ,              // Length of next array
   InputOutput **in_out , // Input/output signal list
   Signal **signals ,     // Existing signals
   int n_conf_comps ,     // Length of next array
   ConfComp *conf_comps , // Confidence compensations
   double prob ,          // Two-tailed probability
   double *excess ,       // Returns 5% excess tail area
   double *toler          // Returns double tail area tolerance
   )
{
   int i, j, k, max_ma, ivar, nouts, ncases, startpos, offset ;
   int casenum, iout, shortest, previous ;
   double *outvars, *optr, *shocks, *wptr, *dptr, pred, err ;
   char msg[84] ;
   InputOutput *ioptr ;
   Signal *sigptr ;
   NPconf *npconf ;

/*
   The number of cases that go into this process is such that no extensions
   past the start or end are needed (except as they naturally occur in
   partially recursive prediction).  Compute the number of fully valid
   cases that can be used for this test.  See Figure 8.3 for an illustration
   of how this formula is derived.  The first fully valid test case is
   offset from the start by maxlag plus 2*max_ma.
*/

   max_ma = 0 ;
   shortest = MAXPOSNUM ;
   for (i=0 ; i<nio ; i++) {          // Pass through all inputs/outputs
      ioptr = in_out[i] ;
      sigptr = signals[ioptr->which] ;
      if (sigptr->n < shortest)              // Find the shortest signal length
         shortest = sigptr->n ;
      if (! ioptr->is_input) {               // Find the max MA lag
         if (ioptr->maxlag > max_ma)
            max_ma = ioptr->maxlag ;
         }
      }

   offset = arma->maxlag + 2 * max_ma ;      // First valid test here
   ncases = shortest - offset - npred + 1 ;  // Number of valid tests

   if ((ncases < 20)  ||  ((int) (0.5 * (1.0 - prob) * ncases) < 1))
      return 2 ;

/*
   Allocate memory for the signals that will be predicted.
   Also allocate the nonparametric confidence object.
*/

   nouts = arma->nout ;       // This many signals predicted

   MEMTEXT ( "ARMACONF: outputs" ) ;

   if (arma->nma)
      outvars = (double *) MALLOC ( 2 * shortest * nouts * sizeof(double) ) ;
   else 
      outvars = (double *) MALLOC ( shortest * nouts * sizeof(double) ) ;

   if (outvars == NULL)
      return -1 ;

   npconf = new NPconf ( nouts , npred , ncases , prob ,
                         n_conf_comps , conf_comps ) ;

   if ((npconf == NULL)  ||  ! npconf->ok) {
      if (npconf != NULL)
         delete npconf ;
      FREE ( outvars ) ;
      return -1 ;
      }


/*
   Initialize the NPconf object by telling it which signals correspond to
   which outputs here.
*/

      ivar = 0 ;                       // Index output variables
      for (i=0 ; i<nio ; i++) {        // Pass through all inputs/outputs
         ioptr = in_out[i] ;
         if (ioptr->is_input)          // If this is an input
            continue ;                 // Ignore it
         for (j=0 ; j<i ; j++) {       // Have we seen this output before?
            if ((! in_out[j]->is_input)  &&
                (in_out[j]->which == ioptr->which))
               break ;
            }
         if (i == j) {  // If this is the first appearance of this output
            sigptr = signals[ioptr->which] ;
            if (npconf->sig_init ( ivar , ioptr->which , sigptr )) {
               FREE ( outvars ) ;
               delete npconf ;
               return -1 ;
               }
            ++ivar ;
            }
         }

/*
   This is the main loop that moves startpos across time, generating
   multiple error measurements at each prediction distance.
*/

   make_progress_window ( "ARMA confidence" ) ;
   previous = 0 ;
   for (startpos=offset ; startpos<ncases+offset ; startpos++) {

      if (! ((startpos-offset) % (ncases / 10))) {
         sprintf ( msg , "%.2lf percent complete",
                   100.0 * (startpos-offset) / ncases ) ;
         write_non_progress ( msg ) ;
         }

#if DEBUG
      printf ( "\nSTARTPOS=%d   previous=%d", startpos, previous ) ;
#endif

/*
   Copy the output signals into the work area.
*/

      ivar = 0 ;                       // Index output variables
      for (i=0 ; i<nio ; i++) {        // Pass through all inputs/outputs
         ioptr = in_out[i] ;
         if (ioptr->is_input)          // If this is an input
            continue ;                 // Ignore it
         for (j=0 ; j<i ; j++) {       // Have we seen this output before?
            if ((! in_out[j]->is_input)  &&
                (in_out[j]->which == ioptr->which))
               break ;
            }
         if (i == j) {  // If this is the first appearance of this output
            sigptr = signals[ioptr->which] ;
            memcpy ( outvars + ivar * shortest + previous ,
                     sigptr->sig + previous ,
                     (startpos + npred - previous) * sizeof(double) ) ;
            ++ivar ;
            }
         }

/*
   This is the inner loop that does predictions based on a fixed startpos
*/

      if (previous < arma->maxlag)   // This happens the first time only
         previous = arma->maxlag ;   // Avoid using undefined AR terms

      for(casenum=previous ; casenum<startpos+npred ; casenum++) {

#if DEBUG
         printf ( "\n  CASENUM=%d", casenum ) ;
#endif

         optr = outvars + casenum ;           // Point to this output
         shocks = optr + shortest * nouts ;   // Shocks go here (if MA terms)

         for (iout=0 ; iout<nouts ; iout++) { // For each predicted output

#if DEBUG
      if (nouts > 1)
         printf ( "\n  out=%d", iout ) ;
      else 
         printf ( "  out=%d", iout ) ;
#endif

            wptr = arma->wts + iout * arma->nvars ; // Weights for this output
            pred = 0.0 ;                      // Will cumulate predicted here
            for (i=0 ; i<nio ; i++) {         // Check all ins and outs
               ioptr = in_out[i] ;
   	         if (ioptr->is_input) {         // If this is an input
                  sigptr = signals[ioptr->which] ; // This is the signal
                  dptr = sigptr->sig ;        // Its values are here
                  for (j=ioptr->minlag ; j<=ioptr->maxlag ; j++) {
                     k = casenum - j ;        // Position of this input case
                     if (k < startpos)        // Common: still within known
                        pred += dptr[k] * *wptr++ ; // Use actual value
                     else if ((k = ioptr->is_other) >= 0)   // Recursive?
                        pred += optr[k*shortest-j] * *wptr++ ; // Use predicted
                     else  // Rare situation: beyond end of unpredicted signal
                        pred += dptr[startpos-1] * *wptr++ ; // Duplicate last

#if DEBUG
                     k = casenum - j ;
                     if (k < startpos)
                        printf ( " (AR past %d=%.3lf)", k, dptr[k] ) ;
                     else if ((k = ioptr->is_other) >= 0)
                        printf ( " (AR rec %d=%.3lf)", casenum-j, optr[k*shortest-j] ) ;
                     else
                        printf ( " (AR dup %d=%.3lf)", startpos-1, dptr[startpos-1] ) ;
#endif

                     }
                  }
               else {      // This is an output
                  k = ioptr->ordinal ;     // Output vector for this MA term
                  for (j=ioptr->minlag ; j<=ioptr->maxlag ; j++) {
                     if (! j)              // Only lags are MA terms
                        continue ;
                     if (casenum - j >= arma->maxlag)   // Shock available?
                        pred += shocks[k*shortest-j] * *wptr++ ; // Use it
                     else                  // Shocks before the first prediction
                        ++wptr ;           // Are assumed zero

#if DEBUG
                     if (casenum - j >= arma->maxlag) { // Shock available?
                        printf ( " (MA%d %d=%.3lf)", k, casenum-j, shocks[k*shortest-j] ) ;
                        if (fabs(shocks[k*shortest-j]) > 10000.0) {
                           printf ( "\a {startpos=%d}", startpos ) ;
                           getch () ;
                           }
                        }
                     else
                        printf ( " (MA%d pre=%d)", k, casenum-j ) ;
#endif

                     }
                  } // This is an output
               } // For all inputs_outputs
            if (! arma->fixed)
               pred += *wptr ;             // Last weight is the constant offset
            if (pred > 1.e30)              // Prevent runaway
               pred = 1.e30 ;
            if (pred < -1.e30)
               pred = -1.e30 ;
            err = optr[iout*shortest] - pred ;

#if DEBUG
            printf ( " --> [%.3lf %.3lf %.3lf]",
                   optr[iout*shortest], pred, err ) ;
#endif

            if (casenum >= startpos) {          // If we are in the future
               npconf->insert ( iout , casenum-startpos ,
                                optr[iout*shortest] , pred ) ;
               optr[iout*shortest] = pred ;     // Record for recursion
               err = 0.0 ;                      // Future shocks are zero

#if DEBUG
            printf ( " S%d", casenum ) ;
#endif

               }
            if (arma->nma)                      // If there are MA terms
               shocks[iout*shortest] = err ;    // Must save the shocks for MA
            } // For all predicted output signals
         } // For casenum

      previous = startpos ;                     // Do not recompute same stuff
      if (! arma->nma)                          // If MA terms, need prev shock
         ++previous ;                           // But if pure AR, no need

      npconf->eval ( startpos , offset ) ;
      } // For startpos

   npconf->conf ( excess , toler ) ;

   destroy_progress_window () ;

   MEMTEXT ( "ARMACONF: outvars, free npconf" ) ;
   FREE ( outvars ) ;
   delete npconf ;

   return 0 ;
}

