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
/*  CONFUSE - All routines related to CLASSIFY confusion                      */
/*                                                                            */
/*    reset_confusion                                                         */
/*    show_confusion                                                          */
/*    save_confusion                                                          */
/*    classify_from_file                                                      */
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

   reset_confusion

--------------------------------------------------------------------------------
*/

void LayerNet::reset_confusion ()
{
   memset ( confusion , 0 , (nout+1) * sizeof(int) ) ;
}

/*
--------------------------------------------------------------------------------

   show_confusion

--------------------------------------------------------------------------------
*/

void LayerNet::show_confusion ()
{
   int i ;
   char *msg ;

   MEMTEXT ( "CONFUSE:show msg" ) ;
   if ((msg = (char *) MALLOC ( (nout+1) * 5 + 11 )) == NULL ) {
      memory_message ( "to SHOW CONFUSION" ) ;
      return ;
      }

   strcpy ( msg , "Confusion:" ) ;
   for (i=0 ; i<nout ; i++)
      sprintf ( msg+5*i+10 , "%5d" , confusion[i] ) ;
   sprintf ( msg+5*nout+10, "%5d", confusion[nout] ) ;
   msg[5*nout+15] = 0 ;
   normal_message ( msg ) ;
   FREE ( msg ) ;
   return ;
}

/*
--------------------------------------------------------------------------------

   save_confusion

--------------------------------------------------------------------------------
*/

void LayerNet::save_confusion ( char *name )
{
   int i ;
   char *msg ;
   FILE *fp ;

   MEMTEXT ( "CONFUSE:save msg" ) ;
   if ((msg = (char *) MALLOC ( (nout+1) * 5 + 1 )) == NULL ) {
      memory_message ( "to SAVE CONFUSION" ) ;
      return ;
      }

/*
   Open the file to which confusion will be written.
   If it already exists, write a newline at its end.
*/

   fp = fopen ( name , "rt" ) ;
   if (fp != NULL) {
      i = 1 ;
      fclose ( fp ) ;
      }
   else
      i = 0 ;

   if ((fp = fopen ( name , "at" )) == NULL) {
      error_message ( "Cannot open SAVE CONFUSION file" ) ;
      FREE ( msg ) ;
      return ;
      }

   if (i)
      fprintf ( fp , "\n" ) ;

/*
   Write confusion
*/

   for (i=0 ; i<nout ; i++)
      sprintf ( msg+5*i , "%5d" , confusion[i] ) ;
   sprintf ( msg+5*nout , "%5d", confusion[nout] ) ;
   msg[5*nout+5] = 0 ;

   fprintf ( fp , "%s", msg ) ;
   fclose ( fp ) ;
   FREE ( msg ) ;
   return ;
}

/*
--------------------------------------------------------------------------------

   classify_from_file

--------------------------------------------------------------------------------
*/

void LayerNet::classify_from_file ( char *name , double thresh )
{
   int i, maxlin, did_any, best ;
   double *inputs, *iptr, maxact ;
   char msg[81], *line, *lptr ;
   FILE *fp ;

/*
   Open the file which contains the data to be classified
*/

   if ((fp = fopen ( name , "rt" )) == NULL) {
      strcpy ( msg , "Cannot open " ) ;
      strcat ( msg , name ) ;
      error_message ( msg ) ;
      return ;
      }

/*
   Allocate for the file lines as read. Conservatively guess length.
   Also allocate for network input vector.
*/

   maxlin = nin_w * 20 + 100 ;
   if (maxlin < 1024)
      maxlin = 1024 ;

   MEMTEXT ( "CONFUSE:line, inputs" ) ;
   line = (char *) MALLOC ( maxlin ) ;
   inputs = (double *) MALLOC ( nin_w * sizeof(double) ) ;

   if ((line == NULL)  ||  (inputs == NULL)) {
      memory_message ( "to classify" ) ;
      fclose ( fp ) ;
      if (line != NULL)
         FREE ( line ) ;
      if (inputs != NULL)
         FREE ( inputs ) ;
      return ;
      }

/*
   Read the file.
*/

   did_any = 0 ;  /* If file runs out on first try, ERROR! */

   for (;;) {  // Endless loop reads until file exhausted

      if ((fgets ( line , maxlin , fp ) == NULL) || (strlen ( line ) < 2)) {
         if ((! did_any)  ||  ferror ( fp )) {
            strcpy ( msg , "Problem reading file " ) ;
            strcat ( msg , name ) ;
            error_message ( msg ) ;
            }
         break ;
         }

      lptr = line ;              // Parse the data from this line
      iptr = inputs ;            // This will be the network inputs
      for (i=0 ; i<nin_w ; i++)
         *iptr++ = ParseDouble ( &lptr ) ;

      did_any = 1 ;              // Flag that at least one found
      trial ( inputs ) ;         // Compute network's outputs

      maxact = -1.e30 ;          // Will keep highest activity here
      best = 0 ;                 // Insurance only (good habit)
      for (i=0 ; i<nout ; i++) { // Find winning output
         if (out[i] > maxact) {
            maxact = out[i] ;
            best = i ;
            }
         }

      if (maxact >= thresh)   // If winner has enough activation
         ++confusion[best] ;  // count it in confusion
      else                    // If too little, throw it
         ++confusion[nout] ;  // in the reject category

      while ((! feof ( fp ))  &&  (line[strlen(line)-1] != '\n'))
         fgets ( line , maxlin , fp ) ; // Line length may exceed maxlin

      if (feof ( fp ))
         break ;
      }  /* Endless loop until a file runs out */

   fclose ( fp ) ;
   MEMTEXT ( "CONFUSE:line, inputs" ) ;
   FREE ( line ) ;
   FREE ( inputs ) ;
}
