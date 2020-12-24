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
/*  EXECUTE - All routines related to AUTO and GENERAL network execution      */
/*                                                                            */
/*  CLASSIF stuff is in CONFUSE.CPP                                           */
/*  But note that EXECUTE is certainly valid in CLASSIF mode.  It outputs     */
/*  neuron activations, just like in the other modes.                         */
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

void PNNet::execute_from_file (
   char *dataname ,  // Input file name
   char *outname     // Output file name
   )
{
   int i, maxlin, did_any ;
   double *inputs, *iptr ;
   char msg[84], *line, *lptr ;
   FILE *fp_in, *fp_out ;

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
   Open the file to which outputs will be written.
   If it already exists, write a newline at its end.
*/

   fp_out = fopen ( outname , "rt" ) ;
   if (fp_out != NULL) {
      did_any = 1 ;
      fclose ( fp_out ) ;
      }
   else
      did_any = 0 ;

   if ((fp_out = fopen ( outname , "at" )) == NULL) {
      strcpy ( msg , "Cannot open output file " ) ;
      strcat ( msg , outname ) ;
      error_message ( msg ) ;
      fclose ( fp_in ) ;
      return ;
      }

   if (did_any)
      fprintf ( fp_out , "\n" ) ;

/*
   Allocate for the file lines as read. Conservatively guess length.
   Also allocate for network input vector.
*/

   maxlin = nin * 20 + 100 ;
   if (maxlin < 1024)
      maxlin = 1024 ;

   MEMTEXT ( "EXECUTE:line, inputs" ) ;
   line = (char *) MALLOC ( maxlin ) ;
   inputs = (double *) MALLOC ( nin * sizeof(double) ) ;

   if ((line == NULL)  ||  (inputs == NULL)) {
      memory_message ( "to execute" ) ;
      fclose ( fp_in ) ;
      fclose ( fp_out ) ;
      if (line != NULL)
         FREE ( line ) ;
      if (inputs != NULL)
         FREE ( inputs ) ;
      return ;
      }

/*
   Read and process the file.
*/

   did_any = 0 ;  /* If file runs out on first try, ERROR! */

   for (;;) {  // Endless loop reads until file exhausted

      if ((fgets ( line , maxlin , fp_in ) == NULL) || (strlen ( line ) < 2)) {
         if ((! did_any)  ||  ferror ( fp_in )) {
            strcpy ( msg , "Problem reading file " ) ;
            strcat ( msg , dataname ) ;
            error_message ( msg ) ;
            }
         break ;
         }

      lptr = line ;           // Parse the data from this line
      iptr = inputs ;         // This will be the network inputs
      for (i=0 ; i<nin ; i++)
         *iptr++ = ParseDouble ( &lptr ) ;

      if (did_any)            // Start each new case on new line
         fprintf ( fp_out , "\n" ) ;

      did_any = 1 ;           // Flag that at least one found
      trial ( inputs ) ;      // Compute network's outputs

      for (i=0 ; i<nout ; i++)
         fprintf ( fp_out , " %.4lf", out[i] ) ;

      while ((! feof ( fp_in ))  &&  (line[strlen(line)-1] != '\n'))
         fgets ( line , maxlin , fp_in ) ; // Line length may exceed maxlin

      if (feof ( fp_in ))
         break ;

      if (user_pressed_escape ()) {   // PeekMessage loop for ESCape & CLOSE
         warning_message ( "Command aborted" ) ;
         break ;
         }

      }  /* Endless loop until a file runs out */

   MEMTEXT ( "EXECUTE:line, inputs" ) ;
   fclose ( fp_in ) ;
   fclose ( fp_out ) ;
   FREE ( line ) ;
   FREE ( inputs ) ;
}
