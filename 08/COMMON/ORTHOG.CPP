/******************************************************************************/
/*                                                                            */
/*  ORTHOG - Data reduction and orthogonalization                             */
/*                                                                            */
/*  Normally, this returns 0.  It returns -1 if insufficient memory.          */
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

/*
--------------------------------------------------------------------------------

   Constructor

   In case of malloc failure, we set 'ok' to zero so the user knows about it.

--------------------------------------------------------------------------------
*/

Orthog::Orthog (
   char *oname ,
   TrainingSet *tptr ,
   MiscParams *misc )
{
   int i, j, tset, ntot, tclass, *classes ;
   double *data, *dptr ;

   strcpy ( name , oname ) ;
   nin = tptr->n_inputs ;
   type = misc->orthog_type ;
   princo = NULL ;
   discrim = NULL ;
   ok = 0 ;

/*
   Keep a record of the training set lags for safety checks
*/

   MEMTEXT ( "Orthog constructor: lags" ) ;
   lags = (unsigned *) MALLOC ( tptr->n_inputs*sizeof(unsigned) ) ;
   if (lags == NULL)
      return ;
   memcpy ( lags , tptr->lags , tptr->n_inputs*sizeof(unsigned) ) ;

/*
   Allocate work arrays
*/

   if (misc->orthog_type == 2)        // If princo of means
      ntot = tptr->n_outputs * tptr->n_inputs ; // We do not keep individual cases
   else
      ntot = tptr->ntrain * tptr->n_inputs ;

   MEMTEXT ( "Orthog constructor: data" ) ;
   data = (double *) MALLOC ( ntot * sizeof(double) ) ;
   if (data == NULL) {
      FREE ( lags ) ;
      return ;
      }

   if (misc->orthog_type == 3) {  // If discriminant function
      MEMTEXT ( "Orthog constructor: classes" ) ;
      classes = (int *) MALLOC ( tptr->ntrain * sizeof(int) ) ;
      if (classes == NULL) {
         FREE ( lags ) ;
         FREE ( data ) ;
         return ;
         }
      }
   else
      classes = NULL ;

/*
   If princo of means, compute mean vectors.
   Otherwise copy the training data to data array.
*/

   if (misc->orthog_type == 2) {  // If princo of means
      for (i=0 ; i<tptr->n_outputs ; i++) {
         for (j=0 ; j<tptr->n_inputs ; j++)
            data[i*tptr->n_inputs + j] = 0.0 ;  // Init sums
         }
      for (tset=0 ; tset<tptr->ntrain ; tset++) {   // Do all samples
         dptr = tptr->data + tptr->size * tset ;    // Point to this sample
         tclass = (int) dptr[tptr->n_inputs] - 1 ;       // Its org 0 class
         for (j=0 ; j<tptr->n_inputs ; j++)              // For all variables
            data[tclass*tptr->n_inputs + j] += dptr[j] ; // Cumulate class' sums
         }
      for (i=0 ; i<tptr->n_outputs ; i++) {              // Divide to get means
         for (j=0 ; j<tptr->n_inputs ; j++)
            data[i*tptr->n_inputs + j] /= tptr->nper[i] ;
         }
      }

   else {
      for (tset=0 ; tset<tptr->ntrain ; tset++) {  // Do all samples
         dptr = tptr->data + tptr->size * tset ;   // Point to this sample
         memcpy ( data + tptr->n_inputs * tset , dptr ,
                  tptr->n_inputs * sizeof(double) ) ;
         if (misc->orthog_type == 3)               // If discriminant function
            classes[tset] = (int) dptr[tptr->n_inputs] - 1 ;
         }
      }

   switch (misc->orthog_type) {
      case 1:
         princo = new PrincoData ( tptr->ntrain , tptr->n_inputs , data ,
            misc->orthog_std , misc->orthog_nfac , misc->orthog_lim , &nfacs ) ;
         break ;
      case 2:
         princo = new PrincoData ( tptr->n_outputs , tptr->n_inputs , data ,
            misc->orthog_std , misc->orthog_nfac , misc->orthog_lim , &nfacs ) ;
         break ;
      case 3:
         discrim = new Discrim ( tptr->ntrain , tptr->n_inputs , tptr->n_outputs , data ,
                   classes , misc->orthog_nfac , misc->orthog_lim , &nfacs ) ;
         break ;
      }

   if (nfacs < 0) {   // Means insufficient memory
      if (princo != NULL) {
         delete princo ;
         princo = NULL ;
         }
      if (discrim != NULL) {
         delete discrim ;
         discrim = NULL ;
         }
      }
   else 
      ok = 1 ;

   MEMTEXT ( "Orthog constructor: data" ) ;
   FREE ( data ) ;
   if (misc->orthog_type == 3) {  // If discriminant function
      MEMTEXT ( "Orthog constructor: classes" ) ;
      FREE ( classes ) ;
      }
}

/*
--------------------------------------------------------------------------------

   Alternate constructor for disk restore

--------------------------------------------------------------------------------
*/

Orthog::Orthog ( char *oname , int onin , int onfacs , int otype ,
                 unsigned *olags )
{
   strcpy ( name , oname ) ;
   nin = onin ;
   type = otype ;
   nfacs = onfacs ;
   princo = NULL ;
   discrim = NULL ;
   ok = 0 ;
   MEMTEXT ( "Orthog alternate constructor: lags" ) ;
   lags = (unsigned *) MALLOC ( nin * sizeof(unsigned) ) ;
   if (lags == NULL)
      return ;
   memcpy ( lags , olags , nin * sizeof(unsigned) ) ;
   ok = 1 ;
}

/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

Orthog::~Orthog ()
{
   if (! ok)
      return ;

   MEMTEXT ( "Orthog destructor (lags, delete princo/discrim)" ) ;
   FREE ( lags ) ;
   if (princo != NULL)
      delete princo ;
   if (discrim != NULL)
      delete discrim ;
}

/*
--------------------------------------------------------------------------------

   transform

   Normally this returns 0.  It returns -1 if insufficient memory.

--------------------------------------------------------------------------------
*/

int Orthog::transform (
   int nio ,              // Length of next array
   InputOutput **in_out , // Input/output signal list
   int *nsigs ,           // Number of signals currently existing
   Signal ***signals ,    // This is them
   MiscParams *misc       // Generated signal names, reduction params
   )
{
   int i, j, k, n, ivar, nouts, ncases, lag ;
   double *data, **outputs, *dptr, *dataptr ;
   Signal **sptr, *sigptr ;
   InputOutput *ioptr ;

/*
   We compute up to the shortest signal.
*/

   ncases = MAXPOSNUM ;
	for (i=0 ; i<nio ; i++) {  // Check all inputs
      if (! in_out[i]->is_input)
         continue ;
      sigptr = (*signals)[in_out[i]->which] ;
      if (sigptr->n < ncases)
         ncases = sigptr->n ;
      }

   nouts = misc->names->nreal ;       // This many signals computed

/*
   Allocate a work array for the input/output data.
   Allocate memory for the signals that will be computed.
*/

   MEMTEXT ( "ORTHOG: data, outputs, signal arrays" ) ;

   data = (double *) MALLOC ( ncases * nin * sizeof(double) ) ;
   outputs = (double **) MALLOC ( nouts * sizeof(double *) ) ;

   if ((data == NULL)  ||  (outputs == NULL)) {
      if (data != NULL)
         FREE ( data ) ;
      if (outputs != NULL)
         FREE ( outputs ) ;
      return -1 ;
      }

   for (i=0 ; i<nouts ; i++) {               // For each computed signal
      outputs[i] = (double *) MALLOC ( ncases * sizeof(double) ) ; // Goes here
      if (outputs[i] == NULL) {
         for (j=0 ; j<i ; j++)
            FREE ( outputs[j] ) ;
         FREE ( data ) ;
         FREE ( outputs ) ;
         return -1 ;
         }
      }

/*
   Count how many of these signals have names not already in use.
   Allocate additional memory for their pointers.
*/

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
      sptr = (Signal **) MALLOC ( nouts * sizeof(Signal *) ) ;

   if (sptr == NULL) {
      for (i=0 ; i<nouts ; i++)
         FREE ( outputs[i] ) ;
      FREE ( data ) ;
      FREE ( outputs ) ;
      return -1 ;
      }
   *signals = sptr ;

/*
   Copy the data into the data array
*/

   dataptr = data ;                           // Will build data array here
   for (k=0 ; k<ncases ; k++) {               // For all cases
      for (i=0 ; i<nio ; i++) {               // Pass through all inputs
         ioptr = in_out[i] ;                  // Signal and lags here
         if (! ioptr->is_input)               // If this is not an input
            continue ;                        // Skip it
         sigptr = (*signals)[ioptr->which] ;  // This is the signal
         dptr = sigptr->sig ;                 // Point to the data
         n = sigptr->n - 1 ;                  // Last of this input
         for (lag=ioptr->minlag ; lag<=ioptr->maxlag ; lag++) {
            j = k - lag ;                     // Get this sample in signal
            if (j < 0)                        // If it is before start
               *dataptr++ = dptr[0] ;         // Use the first sample
            else if (j > n)                   // If beyond the last
               *dataptr++ = dptr[n] ;         // Use the last
            else                              // But under normal conditions
               *dataptr++ = dptr[j] ;         // Use this lagged value
            }
         }
      }

/*
   Compute the factors
*/

   if (princo != NULL)
      princo->factors ( ncases , data ) ;
   else if (discrim != NULL)
      discrim->factors ( ncases , data ) ;

/*
   Copy the computed factors to the outputs
*/

   for (k=0 ; k<ncases ; k++) {               // For all cases
      ivar = 0 ;                              // Locates this signal in outputs
      for (j=0 ; j<misc->names->n ; j++) {    // Check every output name
         if (! misc->names->len[j])           // Some may be NULL
            continue ;                        // Obviously skip them
         if (j < nfacs)                       // Normally we have enough factors
            (outputs[ivar])[k] = data[k*nfacs+j] ; // So output this one
         else                                 // But if more names than factors
            (outputs[ivar])[k] = 0.0 ;        // Just fill with zeros
         ++ivar ;
         }
      } // For all cases


/*
   The final step is to create a new signal for each output.
   If a signal of the same name exists, delete it first.
*/

   ivar = 0 ;
   for (i=0 ; i<misc->names->n ; i++) { // Check all names
      if (! misc->names->len[i])        // Some may be NULL
         continue ;                     // Obviously skip them
      for (j=*nsigs-1 ; j>=0 ; j--) {   // Search existing signals for same name
         if (! strcmp ( misc->names->start[i] , sptr[j]->name ))   // There?
            break ;                     // Quit looking
         }

      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }
      else {                            // This output signal already exists
         MEMTEXT ( "ORTHOG: delete duplicate signal" ) ;
         delete ( sptr[j] ) ;           // Must delete this signal
         }

      MEMTEXT ( "ORTHOG: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , ncases , outputs[ivar] ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;            // This frees the signal
            sptr[j] = NULL ;
            }
         else
            FREE ( outputs[ivar] ) ;    // Manually free it
         for (j=ivar+1 ; j<nouts ; j++) // Free the rest
            FREE ( outputs[j] ) ;
         FREE ( outputs ) ;
         FREE ( data ) ;
         return -1 ;
         }
      ++ivar ;
      } // For all names


   MEMTEXT ( "ORTHOG: data, outputs" ) ;
   FREE ( data ) ;
   FREE ( outputs ) ;

   return 0 ;
}

