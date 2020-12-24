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
/*  TRAIN - All routines related to training                                  */
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

/*
--------------------------------------------------------------------------------

   Constructor

--------------------------------------------------------------------------------
*/

TrainingSet::TrainingSet (
   int out_model ,
   int n_inputs ,
   int n_outputs
   )
{
   outmod = out_model ;

   nin = n_inputs ;
   nout = n_outputs ;

/*
   Compute the size of each training sample.
   CLASSIF output uses a double at the end of each input sample to indicate
   the class membership (0=reject).
   If this is CLASSIFY mode, we also allocate an array that keeps track of
   how many samples are in each class.  This is typically very small, so the
   malloc should not fail.  Nevertheless, we check for that failure when we
   call 'train' to build the training set.  We do the same for the array
   of prior probabilities.
*/

   nper = NULL ;   // Used in CLASSIFY mode to count cases in each class
   priors = NULL ; // Used in CLASSIFY mode for prior probability weights

   if (outmod == OUTMOD_CLASSIFICATION) {
      size = nin + 1 ;
      MEMTEXT ( "TRAIN constructor: nper, priors" ) ;
      nper = (unsigned int *) MALLOC ( nout * sizeof(unsigned) ) ;
      if (nper != NULL)
         memset ( nper , 0 , nout * sizeof(unsigned) ) ;
      priors = (double *) MALLOC ( nout * sizeof(double) ) ;
      }
   else if (outmod == OUTMOD_AUTOASSOCIATION)
      size = nin ;
   else if (outmod == OUTMOD_MAPPING)
      size = nin + nout ;

   ntrain = bufcnt = 0 ;
   data = NULL ;
}


/*
--------------------------------------------------------------------------------

   Destructor

--------------------------------------------------------------------------------
*/

TrainingSet::~TrainingSet ()
{
   if (nper != NULL) {
      MEMTEXT ( "TRAIN: nper" ) ;
      FREE ( nper ) ;
      }
   if (priors != NULL) {
      MEMTEXT ( "TRAIN: priors" ) ;
      FREE ( priors ) ;
      }
   if (data != NULL) {
      MEMTEXT ( "TRAIN: data" ) ;
      FREE ( data ) ;
      }
}

/*
--------------------------------------------------------------------------------

   Assignment operator

--------------------------------------------------------------------------------
*/

void TrainingSet::operator= ( const TrainingSet& tset )
{
   if (this == &tset)
      return ;

   if (nper != NULL) {
      MEMTEXT ( "TRAIN: = nper" ) ;
      FREE ( nper ) ;
      nper = NULL ;
      }
   if (priors != NULL) {
      MEMTEXT ( "TRAIN: = priors" ) ;
      FREE ( priors ) ;
      priors = NULL ;
      }
   if (data != NULL) {
      MEMTEXT ( "TRAIN: = data" ) ;
      FREE ( data ) ;
      data = NULL ;
      }

   outmod = tset.outmod ;
   nin = tset.nin ;
   nout = tset.nout ;
   size = tset.size ;

   ntrain = bufcnt = 0 ;

   if (tset.ntrain) {
      MEMTEXT ( "TRAIN: = data" ) ;
      data = (double *) MALLOC ( tset.ntrain * size * sizeof(double) ) ;
      if (data == NULL)
         return ;
      }

   if (outmod == OUTMOD_CLASSIFICATION) {
      MEMTEXT ( "TRAIN: = nper, priors" ) ;
      nper = (unsigned int *) MALLOC ( nout * sizeof(unsigned) ) ;
      priors = (double *) MALLOC ( nout * sizeof(double) ) ;
      if ((nper == NULL)  ||  (priors == NULL)) {
         if (nper != NULL)
            FREE ( nper ) ;
         if (priors != NULL)
            FREE ( priors ) ;
         if (data != NULL)
            FREE ( data ) ;
         return ;
         }
      memcpy ( nper , tset.nper , nout * sizeof(unsigned) ) ;
      memcpy ( priors , tset.priors , nout * sizeof(double) ) ;
      }

   ntrain = tset.ntrain ;
   if (ntrain)
      memcpy ( data , tset.data , ntrain * size * sizeof(double) ) ;
}


/*
--------------------------------------------------------------------------------

   train - Add members to a training set

--------------------------------------------------------------------------------
*/

void TrainingSet::train (
   char *filename ,   // Full name of file supplying training data
   int outclass ,     // Output class number if CLASSIF output mode
   double prior       // Prior probabilitiy weight
   )
{
   int i, casenum, linelen ;
   unsigned nbuf, tset_bytes, case_limit ;
   double *tptr, *temp ;
   char msg[84], *line, *lptr ;
   FILE *fp ;

/*
   If this is CLASSIFY mode, the constructor allocated an array for holding
   the number of cases in each class.  It did not check for the unlikely event
   of failure of that malloc.  Verify that it went ok.  Ditto for prior.
*/

   if (outmod == OUTMOD_CLASSIFICATION) {
      if ((nper == NULL)  ||  (priors == NULL)) {
         memory_message ( "to allocate for training set" ) ;
         return ;
         }
      priors[outclass-1] = prior ;  // Save prior probability for this class
      }

/*
   We will allocate memory in chunks to save calls to realloc.
   Each chunk will hold nbuf cases.
   The number of training sets is limited by the maximum unsigned int bytes
   which we can allocate.
*/

   tset_bytes = size * sizeof(double) ;
   nbuf = TRAIN_BUF_SIZE / tset_bytes ;
   if (! nbuf) {
      error_message ( "Cases too large.  Increase TRAIN_BUF_SIZE." ) ;
      return ;
      }

/*
   This limit, which is of consequence only for 16-bit versions, is from the
   maximum number of bytes that can be allocated.
*/

   case_limit = ((unsigned) ~0) / (nbuf * tset_bytes) * nbuf ;

   if (ntrain >= case_limit) {
      error_message ( "Too many training sets." ) ;
      return ;
      }

   if ((fp = fopen ( filename , "rt" )) == NULL) {
      sprintf ( msg , "Cannot open %s", filename ) ;
      error_message ( msg ) ;
      return ;
      }

/*
   Conservatively estimate the max length of each file line, then allocate it.
*/

   linelen = (outmod == OUTMOD_MAPPING)  ?  nin + nout  :  nin ;
   linelen = linelen * 32 + 128 ;

   MEMTEXT ( "TRAIN:line" ) ;
   if ((line = (char *) MALLOC ( linelen )) == NULL) {
      memory_message ( "to read training set" ) ;
      fclose ( fp ) ;
      return ;
      }

/*
   If the training set is empty so far, do the initial allocation.
   We allocate for nbuf samples at a time to save calls to REALLOC.
*/

   if (data == NULL) {
      MEMTEXT ( "TRAIN: first block" ) ;
      data = (double *) MALLOC ( nbuf * tset_bytes ) ;
      if (data == NULL) {
         memory_message ( "to read training set" ) ;
         goto FINISH ;
         }
      bufcnt = nbuf ;
      }

/*
   Read the file.
*/

   for (casenum=0 ;; casenum++) {

      if ((fgets ( line , linelen , fp ) == NULL) || (strlen ( line ) < 2)) {
         if (ferror ( fp )  ||  ! casenum) {
            error_message ( "Could not read this file" ) ;
            goto FINISH ;
            }
         else
            break ;   // Normal end of file
         }

      if (! bufcnt--) {  // Allocate a new memory block if needed

         if ((ntrain + nbuf) > case_limit) {
            bufcnt = 0 ;
            error_message ( "Too many training sets." ) ;
            break ;
            }

         MEMTEXT ( "TRAIN: new block" ) ;
         temp = (double *) REALLOC ( data , (ntrain + nbuf) * tset_bytes ) ;
         if (! temp) {
            bufcnt = 0 ;
            memory_message ( "to cumulate this training set" ) ;
            break ;
            }

         data = temp ;
         bufcnt = nbuf - 1 ;
         }

      lptr = line ;                          // Parse the data from this line
      tptr = data + ntrain * size ;          // This sample will go here

      i = nin ;
      while (i--)                            // Inputs
         *tptr++ = ParseDouble ( &lptr ) ;
      if (outmod == OUTMOD_MAPPING) {
         i = nout ;
         while (i--)                         // Outputs
            *tptr++ = ParseDouble ( &lptr ) ;
         }

      while ((! feof ( fp ))  &&  (line[strlen(line)-1] != '\n'))
         fgets ( line , linelen , fp ) ; // Line length may exceed what is needed

      if (outmod == OUTMOD_CLASSIFICATION) { // Save class if CLASSIF output
         *tptr++ = (double) outclass + 0.1 ; // .1 lets us safely truncate
         ++nper[outclass-1] ;                // Count how many in each class
         }

      ++ntrain ;

      if (feof ( fp )) {
         ++casenum ;
         break ;
         }

      }  /* Endless loop until a file runs out */

/*
   Save a little space by shrinking to exactly what we needed
*/

   MEMTEXT ( "TRAIN:final" ) ;
   data = (double *) REALLOC ( data , ntrain * tset_bytes ) ;
   bufcnt = 0 ;

FINISH:
   fclose ( fp ) ;
   MEMTEXT ( "TRAIN:line" ) ;
   FREE ( line ) ;

   sprintf ( msg , "  Read %d, total=%d", casenum, ntrain ) ;
   append_message ( msg ) ;
}
