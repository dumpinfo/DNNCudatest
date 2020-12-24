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
/*  TEST - All routines related to AUTO and GENERAL network testing           */
/*                                                                            */
/*  CLASSIF stuff is in CONFUSE.CPP                                           */
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

void LayerNet::test_from_file ( char *dataname )
{
   int i, n, maxlin, ncases ;
   double *inputs, *iptr, diff, error ;
   char msg[81], *line, *lptr ;
   FILE *fp_in ;

/*
   Compute the number of file line elements
*/

   n = nin_w ;                     // Number of inputs
   if (outmod == OUTMOD_GENERAL)   // That's all if AUTO
      n += nout_n ;                // And outputs

/*
   Open the file that contains the data to be classified
*/

   if ((fp_in = fopen ( dataname , "rt" )) == NULL) {
      strcpy ( msg , "Cannot open input data file " ) ;
      strcat ( msg , dataname ) ;
      error_message ( msg ) ;
      return ;
      }

/*
   Allocate for the file lines as read. Conservatively guess length.
   Also allocate for network input vector.
*/

   maxlin = n * 20 + 100 ;
   if (maxlin < 1024)
      maxlin = 1024 ;

   MEMTEXT ( "TEST:line, inputs" ) ;
   line = (char *) MALLOC ( maxlin ) ;
   inputs = (double *) MALLOC ( n * sizeof(double) ) ;

   if ((line == NULL)  ||  (inputs == NULL)) {
      memory_message ( "to test" ) ;
      fclose ( fp_in ) ;
      if (line != NULL)
         FREE ( line ) ;
      if (inputs != NULL)
         FREE ( inputs ) ;
      return ;
      }

/*
   Read and process the file.
*/

   ncases = 0 ;   // Count cases in file
   error = 0.0 ;  // Will sum error here

   for (;;) {  // Endless loop reads until file exhausted

      if ((fgets ( line , maxlin , fp_in ) == NULL) || (strlen ( line ) < 2)) {
         if ((! ncases)  ||  ferror ( fp_in )) {
            strcpy ( msg , "Problem reading file " ) ;
            strcat ( msg , dataname ) ;
            error_message ( msg ) ;
            }
         break ;
         }

      lptr = line ;             // Parse the data from this line
      iptr = inputs ;           // This will be the network inputs
      for (i=0 ; i<nin_w ; i++) // Get inputs
         *iptr++ = ParseDouble ( &lptr ) ;

      if (outmod == OUTMOD_GENERAL) {  // Also get outputs if GENERAL
         for (i=0 ; i<nout_n ; i++)
            *iptr++ = ParseDouble ( &lptr ) ;
         }

      ++ncases ;            // Count this case
      trial ( inputs ) ;    // Compute network's outputs

      if (outmod == OUTMOD_AUTO) {
         for (i=0 ; i<nin_w ; i++) {
            diff = inputs[i] - out[i] ;
            error += diff * diff ;
            }
         }
      else {   // Must be GENERAL
         for (i=0 ; i<nout_n ; i++) {
            diff = inputs[i+nin_w] - out[i] ;
            error += diff * diff ;
            }
         }

      while ((! feof ( fp_in ))  &&  (line[strlen(line)-1] != '\n'))
         fgets ( line , maxlin , fp_in ) ; // Line length may exceed maxlin

      if (feof ( fp_in ))
         break ;
      }  /* Endless loop until a file runs out */

   fclose ( fp_in ) ;
   MEMTEXT ( "TEST:line, inputs" ) ;
   FREE ( line ) ;
   FREE ( inputs ) ;

   error /= (double) ncases * (double) nout ;
   printf ( "\nMean Squared Error = %lf    RMS = %lf", error, sqrt(error) ) ;
}
