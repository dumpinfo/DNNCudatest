/******************************************************************************/
/*                                                                            */
/*  NET_PRED - Predict using a neural network                                 */
/*                                                                            */
/*  Normally, this returns 0.  It returns -1 if insufficient memory,          */
/*  1 if user pressed ESCape.                                                 */
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

int net_pred (
   int n_inputs ,         // Number of input variables
   int npred ,            // Number of cases to generate
   Network *net ,         // Use this network
   MiscParams *misc ,     // Mainly for generated signal names
   int n_inputs_outputs , // Length of next array
   InputOutput **in_out , // Input/output signal list
   int *nsigs ,           // Number of signals currently existing
   Signal ***signals      // This is them
   )
{
   int i, j, k, n, ivar, nvars, casenum, lag, lead, user_quit ;
   int *in_length, startpos ;
   double **outputs, **inlist, *dptr, *in_vector, *inptr ;
   char msg[84] ;
   Signal **sptr, *sigptr ;
   InputOutput *ioptr ;

/*
   This array will be used to flag input sources.  In the case that
   a signal is an input only, its entry will be NULL.  But if a user
   is using an input as an output also, the output pointer will go here.
*/

   MEMTEXT ( "NET_PRED: inlist" ) ;
   inlist = (double **) MALLOC ( n_inputs_outputs * sizeof(double*) ) ;
   if (inlist == NULL)
      return -1 ;

/*
   This is used for the network's input vector
*/

   MEMTEXT ( "NET_PRED: in_vector" ) ;
   in_vector = (double *) MALLOC ( n_inputs * sizeof(double) ) ;
   if (in_vector == NULL) {
      FREE ( inlist ) ;
      return -1 ;
      }

/*
   Allocate memory for the signals that will be predicted.
   Count how many of these signals have names not already in use.
   Allocate additional memory for their pointers.
*/

   nvars = misc->names->nreal ;              // This many signals predicted

   MEMTEXT ( "NET_PRED: outputs, signal arrays" ) ;
   outputs = (double **) MALLOC ( nvars * sizeof(double *) ) ;
   in_length = (int *) MALLOC ( nvars * sizeof(int) ) ;
   if ((outputs == NULL)  ||  (in_length == NULL)) {
      FREE ( inlist ) ;
      FREE ( in_vector ) ;
      if (outputs != NULL)
         FREE ( outputs ) ;
      if (in_length != NULL)
         FREE ( in_length ) ;
      return -1 ;
      }

   for (i=0 ; i<nvars ; i++) {               // For each predicted signal
      in_length[i] = 0 ;                     // Length of common input (if any)
      outputs[i] = (double *) MALLOC ( npred * sizeof(double) ) ; // Goes here
      if (outputs[i] == NULL) {
         for (j=0 ; j<i ; j++)
            FREE ( outputs[j] ) ;
         FREE ( outputs ) ;
         FREE ( inlist ) ;
         FREE ( in_vector ) ;
         FREE ( in_length ) ;
         return -1 ;
         }
      for (j=0 ; j<npred ; j++)              // Initialize this output signal
         (outputs[i])[j] = 0.0 ;             // To prevent NAN problems
      }

   if (*nsigs) {                             // If signals already exist
      ivar = *nsigs ;                        // This many signals so far
      sptr = *signals ;                      // Array of pointers to them
      for (i=0 ; i<misc->names->n ; i++) {   // Check every new name
         if (! misc->names->len[i])          // Some may be NULL
            continue ;                       // Obviously skip them
         for (j=*nsigs-1 ; j>=0 ; j--) {     // Check every existing signal
            if (! strcmp ( misc->names->start[i] , sptr[j]->name )) // There?
               break ;                       // If found, quit looking
            }
         if (j < 0)                          // Means not there
            ++ivar ;                         // So count this new entry
         }
      sptr = (Signal **) REALLOC ( sptr , ivar * sizeof(Signal *) ) ;
      }
   else
      sptr = (Signal **) MALLOC ( nvars * sizeof(Signal *) ) ;

   if (sptr == NULL) {
      for (i=0 ; i<nvars ; i++)
         FREE ( outputs[i] ) ;
      FREE ( outputs ) ;
      FREE ( inlist ) ;
      FREE ( in_vector ) ;
      FREE ( in_length ) ;
      return 1 ;
      }
   *signals = sptr ;

/*
   Some users may want to predict signals that also serve as input.
   This necessitates a little preparation.  Run through the input signals.
   For each that is also an output, flag that fact by storing its output
   pointer.  Otherwise store a NULL.
   Also, store the length of the input signal in in_length and copy that
   input signal to the output.  When that output is computed, we will not
   allow overwriting of input values.  So that we can avoid computing
   outputs that will not be used, keep track of the minimum starting
   position.
   Finally, note that this whole section is meaningless in CLASSIFICATION mode.
   Nevertheless, we still want inlist to be all NULL, so put the test here.
*/

   startpos = npred ;                        // Avoids overwriting inputs
   for (i=0 ; i<n_inputs_outputs ; i++) {
      inlist[i] = NULL ;                     // Assume this is not recursive
      if (net->output_mode == OUTMOD_CLASSIFICATION) // In CLASSIFICATION mode
         continue ;                          // The rest is meaningless
      ioptr = in_out[i] ;
      if (! ioptr->is_input)                 // If this is an output list entry
         continue ;                          // Ignore it
      sigptr = (*signals)[ioptr->which] ;    // This input signal
      ivar = 0 ;                             // Locates this signal in outputs
      for (j=0 ; j<misc->names->n ; j++) {   // Check every output name
         if (! misc->names->len[j])          // Some may be NULL
            continue ;                       // Obviously skip them
         if (! strcmp ( misc->names->start[j] , sigptr->name )) { // This one?
            inlist[i] = outputs[ivar] ;      // This input is here among outputs
            n = (npred < sigptr->n) ? npred : sigptr->n ; // Copy input to out
            in_length[ivar] = n ;            // Will protect the inputs here
            memcpy ( outputs[ivar] , sigptr->sig , n * sizeof(double) ) ;
            if ((n - net->leads[j]) < startpos) // Must protect this many
               startpos = n - net->leads[j] ;   // But start soon enough for all
            break ;
            }
         ++ivar ;                            // Keep track of output location
         }
      }

/*
   We are almost done computing startpos, the place to start processing.
   It is possible that the above computation produced a negative value.
   Also, if there is even one output that is not recursive, we must start
   right at the beginning.
*/

   for (i=0 ; i<nvars ; i++) {   // For every output variable
      if (in_length[i] == 0)     // If it is not also an input
         startpos = 0 ;          // We must catch it from the start
      }

   if (startpos < 0)
      startpos = 0 ;

/*
   At last we can generate the signals
*/

   make_progress_window ( "Network prediction" ) ;
   user_quit = 0 ;
   for (casenum=startpos ; casenum<npred ; casenum++) {// Start late as possible
#if DEBUG
      printf ( "\ncasenum=%2d", casenum ) ;
#endif
      inptr = in_vector ;                     // Will build input vector here
      for (i=0 ; i<n_inputs_outputs ; i++) {  // Pass through all inputs
         ioptr = in_out[i] ;                  // Signal and lags here
         if (! ioptr->is_input)               // If this is not an input
            continue ;                        // Skip it
         if (inlist[i] != NULL) {             // If this input is also an output
            dptr = inlist[i] ;                // Get its address
            n = npred - 1 ;                   // Last of this input
#if DEBUG
            printf ( "\n  REC%d n=%d", i, n ) ;
#endif
            }
         else {                               // Else we must get the signal
            sigptr = (*signals)[ioptr->which] ; // This is the signal
            dptr = sigptr->sig ;              // Get the data
            n = sigptr->n - 1 ;               // Last of this input
#if DEBUG
            printf ( "\n  INP%d n=%d", i, n ) ;
#endif
            }
         for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++) {
            j = casenum - lag ;               // Get this sample in signal
            if (j < 0)                        // If it is before start
               *inptr++ = dptr[0] ;           // Use the first sample
            else if (j > n)                   // If beyond the last
               *inptr++ = dptr[n] ;           // Use the last
            else                              // But under normal conditions
               *inptr++ = dptr[j] ;           // Use this lagged value
#if DEBUG
            printf ( " (%d=%.2lf)", j, *(inptr-1) ) ;
#endif
            }
         }

      net->trial ( in_vector ) ;              // Evaluate network for input

      ivar = 0 ;                              // Locates this signal in outputs
      for (j=0 ; j<misc->names->n ; j++) {    // Check every output name
         if (! misc->names->len[j])           // Some may be NULL
            continue ;                        // Obviously skip them
         if (j >= net->n_outputs)                  // Careless user may give too
            break ;                           // many names, so check
         if (net->output_mode == OUTMOD_CLASSIFICATION) // In CLASSIFICATION mode
            lead = 0 ;                        // Lead is always zero
         else                                 // In MAPPING mode
            lead = net->leads[j] ;            // Lead was recorded when trained
         k = casenum + lead ;                 // It goes in this time slot
#if DEBUG
         printf ( " (j=%d iv=%d k=%d out=%.2lf", j, ivar, k, net->out[j] ) ; /*!!!!!!*/
#endif
         if ((k < npred) && (k >= in_length[ivar])) // Not past end, protect in
            (outputs[ivar])[k] = net->out[j] ; // Output this variable
#if DEBUG
         else
            printf ( " X" ) ;
#endif
         ++ivar ;
         }

      if (user_pressed_escape ()) {
         user_quit = 1 ;
         break ;
         }
#if DEBUG > 2
   getch () ;
#endif

      if (! ((casenum-startpos) % ((npred-startpos) / 10))) {
         sprintf ( msg , "%.2lf percent complete",
            100.0 * (casenum-startpos) / (npred-startpos) ) ;
         write_non_progress ( msg ) ;
         }
      } // For all npred cases
#if DEBUG > 1
   getch () ;
#endif

/*
   The final step is to create a new signal for each output.
   If a signal of the same name exists, just replace the data.
*/

   destroy_progress_window () ;

   ivar = 0 ;
   for (i=0 ; i<misc->names->n ; i++) { // Check all names
      if (! misc->names->len[i])        // Some may be NULL
         continue ;                     // Obviously skip them
      for (j=*nsigs-1 ; j>=0 ; j--) {   // Search existing signals for same name
         if (! strcmp ( misc->names->start[i] , sptr[j]->name ))   // There?
            break ;                     // Yes, so quit looking
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         MEMTEXT ( "NET_PRED: new Signal" ) ;
         sptr[j] = new Signal ( misc->names->start[i] , npred , outputs[ivar] ) ;
         if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
            if (sptr[j] != NULL) {
               delete sptr[j] ;            // This frees the signal
               sptr[j] = NULL ;
               }
            else
               FREE ( outputs[ivar] ) ;    // Manually free it
            for (j=ivar+1 ; j<nvars ; j++) // Free the rest
               FREE ( outputs[j] ) ;
            FREE ( inlist ) ;
            FREE ( in_vector ) ;
            FREE ( in_length ) ;
            return -1 ;
            }
         }
      else {                            // This output signal already exists
         MEMTEXT ( "NET_PRED: replace signal" ) ;
         if (in_length[ivar])           // Is it a recursive input?
            sptr[j]->replace ( npred , in_length[ivar] , outputs[ivar] ) ;
         else {                         // Also protect signals in OUTPUT list
            n = 0 ;                     // If not there, will protect 0 cases
            for (k=0 ; k<n_inputs_outputs ; k++) {
               ioptr = in_out[k] ;
               if ((! ioptr->is_input)  &&  (ioptr->which == j)) { // There?
                  n = sptr[j]->n ;
                  break ;
                  }
               }

            sptr[j]->replace ( npred , n , outputs[ivar] ) ;
            }
         }

      ++ivar ;
      } // For all names


   MEMTEXT ( "NET_PRED: outputs, inlist, in_vector, in_length" ) ;
   FREE ( outputs ) ;
   FREE ( inlist ) ;
   FREE ( in_vector ) ;
   FREE ( in_length ) ;

   return user_quit ;
}
