/******************************************************************************/
/*                                                                            */
/*  NET_CONF - Compute prediction confidence for neural networks              */
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

int net_conf (
   int n_inputs ,         // Number of input variables
   int npred ,            // Number of future predictions
   Network *net ,         // Use this network
   MiscParams *misc ,     // Mainly for signal names
   int n_inputs_outputs , // Length of next array
   InputOutput **in_out , // Input/output signal list
   int nsigs ,            // Number of signals currently existing
   Signal **signals ,     // This is them
   int n_conf_comps ,     // Length of next array
   ConfComp *conf_comps , // Confidence compensations
   double *excess ,       // Returns 5% excess tail area
   double *toler          // Returns double tail area tolerance
   )
{
   int i, j, k, ivar, nvars, casenum, lag, lead, maxlag, maxlead, minlead ;
   int startpos, ncases, shortest, offset, user_quit, endcase ;
   double **outputs, **inlist, *in_vector, *inptr ;
   char msg[84] ;
   Signal *sigptr ;
   InputOutput *ioptr ;
   NPconf *npconf ;

/*
   The number of cases that go into this process is such that no extensions
   past the start or end are needed (except as they naturally occur in
   partially recursive prediction).  Compute the number of fully valid
   cases that can be used for this test.  See Figure 8.2 for an illustration
   of how this formula is derived.  The first fully valid test case is
   offset from the start by maxlag+maxlead.

   These limits are computed in three steps.  The INPUT LIST determines
   the max lag and the shortest signal.  The NAMES list also determines
   the shortest signal.  The network determines the min and max lags.

   For each starting position, we will work with npred predicted outputs.
   These have a dual use.  They are initialized to the true values so that
   they are the reference for errors.  Then the true values are replaced
   by the predictions for recursive use as inputs.
*/

   maxlag = maxlead = 0 ;
   shortest = minlead = MAXPOSNUM ;
   for (i=0 ; i<n_inputs_outputs ; i++) {  // Pass through all inputs/outputs
      ioptr = in_out[i] ;
      if (! ioptr->is_input)               // Outputs are ignored
         continue ;
      sigptr = signals[ioptr->which] ;
      if (sigptr->n < shortest)            // Find the shortest signal length
         shortest = sigptr->n ;
      if (ioptr->maxlag > maxlag)
         maxlag = ioptr->maxlag ;
      }

   for (j=0 ; j<net->n_outputs ; j++) {
      lead = net->leads[j] ;             // Lead was recorded when trained
      if (lead > maxlead)                // Find the max output lead
         maxlead = lead ;
      if (lead < minlead)                // And the min
         minlead = lead ;
      }

   for (j=0 ; j<misc->names->n ; j++) {   // Check every output name
      if (! misc->names->len[j])          // Some may be NULL
         continue ;                       // Obviously skip them
      for (i=0 ; i<nsigs ; i++) {         // Find this signal
         sigptr = signals[i] ;
         if (! strcmp ( misc->names->start[j] , sigptr->name )) {
            if (sigptr->n < shortest)     // Find the shortest signal length
               shortest = sigptr->n ;
            break ;
            }
         }
      }

   offset = maxlag + maxlead ;               // First fully valid test is here
   ncases = shortest - offset - npred + 1 ;  // Number of fully valid tests
   nvars = misc->names->nreal ;              // This many signals predicted

   if ((ncases < 20)  ||  ((int) (0.5 * (1.0 - misc->conf_prob) * ncases) < 1))
      return 2 ;

   npconf = new NPconf ( nvars , npred , ncases , misc->conf_prob ,
                         n_conf_comps , conf_comps ) ;

   if ((npconf == NULL)  ||  ! npconf->ok) {
      if (npconf != NULL)
         delete npconf ;
      return -1 ;
      }

/*
   Initialize the NPconf object by telling it which signals correspond to
   which outputs.
*/

   ivar = 0 ;                             // Locates this signal in outputs
   for (j=0 ; j<misc->names->n ; j++) {   // Check every output name
      if (! misc->names->len[j])          // Some may be NULL
         continue ;                       // Obviously skip them
      for (i=0 ; i<nsigs ; i++) {         // Find this signal
         sigptr = signals[i] ;
         if (! strcmp ( misc->names->start[j] , sigptr->name )) {
            if (npconf->sig_init ( ivar , i , sigptr )) { // Init NPconf object
               delete npconf ;
               return -1 ;
               }
            break ;
            }
         }
      ++ivar ;                            // Keep track of output location
      }

/*
   This array will be used to flag input sources.  In the case that a
   signal is an input only, its entry will be NULL.  But if the user
   is using an input as an output also, the output pointer will go here.
*/

   MEMTEXT ( "NET_CONF: inlist, in_vector, outputs" ) ;
   inlist = (double **) MALLOC ( n_inputs_outputs * sizeof(double*) ) ;
   if (inlist == NULL) {
      delete npconf ;
      return -1 ;
      }

/*
   This is used for the network's input vector
*/

   in_vector = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   if (in_vector == NULL) {
      FREE ( inlist ) ;
      delete npconf ;
      return -1 ;
      }

/*
   Allocate memory for the signals that will be predicted.
   For each startpos, we keep npred predictions.
*/

   outputs = (double **) MALLOC ( nvars * sizeof(double *) ) ;
   if (outputs == NULL) {
      FREE ( inlist ) ;
      FREE ( in_vector ) ;
      delete npconf ;
      return -1 ;
      }

   for (i=0 ; i<nvars ; i++) {               // For each predicted signal
      outputs[i] = (double *) MALLOC ( npred * sizeof(double) ) ;
      if (outputs[i] == NULL) {
         for (j=0 ; j<i ; j++)
            FREE ( outputs[j] ) ;
         FREE ( outputs ) ;
         FREE ( inlist ) ;
         FREE ( in_vector ) ;
         delete npconf ;
         return -1 ;
         }
      }

/*
   We often want to predict (recursively) signals that also serve as input.
   This necessitates a little preparation.  Run through the input signals.
   For each that is also an output, flag that fact by storing its output
   pointer.  Otherwise store a NULL as a flag to get the actual signal.
*/

   for (i=0 ; i<n_inputs_outputs ; i++) {    // Check all inputs
      inlist[i] = NULL ;                     // Assume this is not recursive
      ioptr = in_out[i] ;                    // All of these should be inputs
      if (! ioptr->is_input)                 // If this is an output list entry
         continue ;                          // Ignore it, as it is meaningless
      sigptr = signals[ioptr->which] ;       // This is the input signal
      ivar = 0 ;                             // Locates this signal in outputs
      for (j=0 ; j<misc->names->n ; j++) {   // Check every output name
         if (! misc->names->len[j])          // Some may be NULL
            continue ;                       // Obviously skip them
         if (! strcmp ( misc->names->start[j] , sigptr->name )) { // This one?
            inlist[i] = outputs[ivar] ;      // This input is here among outputs
            break ;                          // So flag its recusive nature
            }                                // No need to keep looking
         ++ivar ;                            // Keep track of output location
         }
      }

/*
   This is the main outer loop.
   Each 'outputs' vector (npred long) starts at 'startpos'.
   For each new starting position, the first step is to set the recursive
   output vectors to their true values.  As the predictions are done, these
   will be the reference for keeping track of the errors.  Also, the true
   values will be overwritten with the predicted values for subsequent
   recursive use as inputs.  It is assumed that correct values are known
   up to (but not including) startpos.  Everything from startpos onward
   is a prediction.
*/

   make_progress_window ( "Network confidence" ) ;
   user_quit = 0 ;
   for (startpos=offset ; startpos<ncases+offset ; startpos++) {

      if (! ((startpos-offset) % (ncases / 10))) {
         sprintf ( msg , "%.2lf percent complete",
                   100.0 * (startpos-offset) / ncases ) ;
         write_non_progress ( msg ) ;
         }

#if DEBUG
      printf ( "\n\nSTARTPOS=%d", startpos ) ;
#endif

      ivar = 0 ;                             // Locates this signal in outputs
      for (j=0 ; j<misc->names->n ; j++) {   // Check every output name
         if (! misc->names->len[j])          // Some may be NULL
            continue ;                       // Obviously skip them
         for (i=0 ; i<nsigs ; i++) {         // Find this signal so we can
            sigptr = signals[i] ;            // Copy it into output vector
            if (! strcmp ( misc->names->start[j] , sigptr->name )) {
#if DEBUG
               printf ( " (%s=%d)", sigptr->name, ivar ) ;
#endif
               memcpy ( outputs[ivar] , sigptr->sig+startpos ,
                        npred * sizeof(double) ) ;
               break ;                       // We copy npred cases starting
               }                             // at startpos, the first
            }                                // 'unknown' output
         ++ivar ;                            // Keep track of output location
         }

/*
   We are ready to do the predictions.  Everything before startpos is known.
   Start predicting at startpos-maxlead so we bag the longest lead.
   Go as far as we need to get the last of the npred predictions for the
   signal with the shortest lead.
*/

      endcase = startpos + npred - minlead ;

      for (casenum=startpos-maxlead ; casenum<endcase ; casenum++) { // Predicts
#if DEBUG
         printf ( "\ncasenum=%d", casenum ) ;
#endif
         inptr = in_vector ;                     // Will build input vector here
         for (i=0 ; i<n_inputs_outputs ; i++) {  // Pass through all inputs
            ioptr = in_out[i] ;                  // Signal and lags here
            if (! ioptr->is_input)               // If this is not an input
               continue ;                        // Skip it (irrelevant)
            for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++) {
               k = casenum - lag ;               // This ordinal case
               if (k < startpos)                 // If still in known cases
                  *inptr++ = (signals[ioptr->which])->sig[k] ; // Use true value
               else if (inlist[i] != NULL)       // Beyond.  If recursive
                  *inptr++ = (inlist[i])[k-startpos] ; // Use this prediction
               else                              // Rare.  Dup final point.
                  *inptr++ = (signals[ioptr->which])->sig[startpos-1] ;
#if DEBUG
               if (k < startpos)                 // If still in known cases
                  printf ( " in%d=%.3lf", k, (signals[ioptr->which])->sig[k] ) ;
               else if (inlist[i] != NULL)       // Beyond.  If recursive
                  printf ( " rec%d=%.3lf", k, (inlist[i])[k-startpos] ) ;
               else                              // Rare.  Dup final point.
                  printf ( " dup%d=%.3lf", k, (signals[ioptr->which])->sig[startpos-1] ) ;
#endif
               }
            }

         net->trial ( in_vector ) ;            // Evaluate network for input

#if DEBUG > 1
         printf ( "\nIN: " ) ;
         for (i=0 ; i<net->n_inputs ; i++)
            printf ( "  %lf", in_vector[i] ) ;
         printf ( "\nOT: " ) ;
         for (i=0 ; i<net->n_outputs ; i++)
            printf ( "  %lf", net->out[i] ) ;
#endif

         ivar = 0 ;                            // Locates this signal in outputs
         for (j=0 ; j<misc->names->n ; j++) {  // Check every output name
            if (! misc->names->len[j])         // Some may be NULL
               continue ;                      // Obviously skip them
            if (j >= net->n_outputs)                // Careless user may give too
               break ;                         // many names, so check
            lead = net->leads[j] ;             // Lead was recorded when trained
            k = casenum - startpos + lead ;    // It goes in this time slot
#if DEBUG
            printf ( " k=%d", k ) ;
#endif
            if ((k >= 0)  &&  (k < npred)) {   // In the window of npred tests?
#if DEBUG
               printf ( " (%.4lf %.4lf)", (outputs[ivar])[k], net->out[j] ) ;
#endif
               npconf->insert ( ivar , k , (outputs[ivar])[k] , net->out[j] ) ;
               (outputs[ivar])[k] = net->out[j] ; // Keep this prediction
               }
            ++ivar ;
            } // For all outputs

         if (user_pressed_escape ()) {
            user_quit = 1 ;
            break ;
            }

         } // For all npred cases

      npconf->eval ( startpos , offset ) ;

      if (user_quit)
         break ;
      } // For all startpos starting positions

   npconf->conf ( excess , toler ) ;

   destroy_progress_window () ;

   MEMTEXT ( "NET_CONF: outputs, inlist, in_vector, free npconf" ) ;
   for (i=0 ; i<nvars ; i++)
      FREE ( outputs[i] ) ;
   FREE ( outputs ) ;
   FREE ( inlist ) ;
   FREE ( in_vector ) ;
   delete npconf ;

   return user_quit ;
}
