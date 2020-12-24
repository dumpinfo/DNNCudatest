/******************************************************************************/
/*                                                                            */
/*  READSIG - Read univariate or multivariate signals                         */
/*                                                                            */
/*  Normally, this returns 0.  It returns 1 if an error occurred.             */
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

#define READSIG_BUF 4096

int readsig ( MiscParams *misc , char *filename , int *nsigs ,
              Signal ***signals , char *error )
{
   int i, j, k, ivar, nvars, bufcnt, linelen, ncases ;
   char *line, *lptr ;
#if DEBUG
   char msg[256] ; /*!!!!!!*/
#endif
   double *data, *temp, *dptr, x ;
   FILE *fp ;
   Signal **sptr ;

   *error = 0 ;  // Flag no error
   nvars = misc->names->nreal ;

   if (! nvars) {
      strcpy ( error , "No signal names specified" ) ;
      return -1 ;
      }

   if ((fp = fopen ( filename , "rt" )) == NULL) {
      sprintf ( error , "Cannot open %s", filename ) ;
      return -1 ;
      }

   linelen = 32 * misc->names->n + 1024 ;
   MEMTEXT ( "READSIG: data, line" ) ;
   data = (double *) MALLOC ( READSIG_BUF * nvars * sizeof(double) ) ;
   line = (char *) MALLOC ( linelen ) ;

   if ((data == NULL)  ||  (line == NULL)) {
      strcpy ( error , "Insufficient memory to read signal" ) ;
      fclose ( fp ) ;
      if (data == NULL)
         FREE ( data ) ;
      if (line == NULL)
         FREE ( line ) ;
      return -1 ;
      }

   bufcnt = READSIG_BUF ;

/*
   Read the file.
*/

   ncases = 0 ;
   for (;;) {

      if ((fgets ( line , linelen , fp ) == NULL) || (strlen ( line ) < 2)) {
         if (ferror ( fp )  ||  ! ncases) {
            strcpy ( error , "Could not read this file" ) ;
            FREE ( data ) ;
            goto FINISH ;
            }
         else
            break ;       // Normal end of file
         }

#if DEBUG
      MEMTEXT ( line ) ;
      printf ( "\nCase %d (%s) = %d", ncases, line, strlen(line) ) ;
#endif

      if (! bufcnt--) {  // Allocate a new memory block if needed

         MEMTEXT ( "READSIG: new block" ) ;
         temp = (double *) REALLOC ( data ,
                           (ncases + READSIG_BUF) * nvars * sizeof(double) ) ;
         if (! temp) {
            strcpy ( error , "Insufficient memory to read signal file" ) ;
            FREE ( data ) ;
            goto FINISH ;
            }

         data = temp ;
         bufcnt = READSIG_BUF - 1 ;
         }

      lptr = line ;                        // Parse the data from this line
      dptr = data + ncases * nvars ;       // This sample will go here

      for (i=0 ; i<misc->names->n ; i++) { // Parse out for each string
         x = ParseDouble ( &lptr ) ;       // The data is here
#if DEBUG
         sprintf ( msg , "%lf", x ) ;
         MEMTEXT ( msg ) ;
#endif
         if (misc->names->len[i])          // Only if this name non-NULL
            *dptr++ = x ;                  // Keep it
         }

      ++ncases ;                           // Count cases read

      while ((! feof ( fp ))  &&  (line[strlen(line)-1] != '\n'))
         fgets ( line , linelen , fp ) ; // Input file lines may be extra long

      if (feof ( fp ))
         break ;

      }  /* Endless loop until file runs out */

   if (! ncases) {
      strcpy ( error , "This file had no cases" ) ;
      FREE ( data ) ;
      goto FINISH ;
      }

/*
   Count how many of these signals have names not already in use.
   Then allocate additional memory for their pointers.
*/

   MEMTEXT ( "READSIG: signals array" ) ;
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
      FREE ( data ) ;
      goto FINISH ;
      }
   *signals = sptr ;

/*
   Now create new signals for each variable.
   If a signal of the same name exists, delete it first.
   If there is only one variable, we can simply use 'data' for it.
   Otherwise we must unscramble the interlaced data.
*/

   ivar = 0 ;                           // Current variable being processed
   for (i=0 ; i<misc->names->n ; i++) { // Check all names
      if (! misc->names->len[i])        // Some may be NULL
         continue ;                     // Obviously skip them
      for (j=*nsigs-1 ; j>=0 ; j--) {   // Search existing signals for same name
         if (! strcmp ( misc->names->start[i] , sptr[j]->name )) {  // There?
            MEMTEXT ( "READSIG: delete duplicate signal" ) ;
            delete ( sptr[j] ) ;        // If so, delete this signal
            break ;                     // And quit looking
            }
         }
      if (j < 0) {                      // Means new, unique name
         j = *nsigs ;                   // Tack it onto end of signal array
         ++*nsigs ;                     // And count it
         }

      MEMTEXT ( "READSIG: temp signal" ) ;
      if (nvars == 1)   // In this case, no unscrambling, so simple copy OK
         temp = (double *) REALLOC ( data , ncases * sizeof(double) ) ;
      else {   // Otherwise signals are interlaced, so must extract
         temp = (double *) MALLOC ( ncases * sizeof(double) ) ;
         if (temp == NULL) {
            strcpy ( error , "Insufficient memory to read signal" ) ;
            break ;
            }
         for (k=0 ; k<ncases ; k++)
            temp[k] = data[k*nvars+ivar] ;
         }

      MEMTEXT ( "READSIG: new Signal" ) ;
      sptr[j] = new Signal ( misc->names->start[i] , ncases , temp ) ;
      if ((sptr[j] == NULL)  ||  ! sptr[j]->n) {
         if (sptr[j] != NULL) {
            delete sptr[j] ;
            sptr[j] = NULL ;
            }
         strcpy ( error , "Insufficient memory to read signal" ) ;
         break ;
         }
      ++ivar ;
      } // For all names

   if (nvars != 1) {
      MEMTEXT ( "READSIG: data" ) ;
      FREE ( data ) ;
      }

FINISH:
   fclose ( fp ) ;
   MEMTEXT ( "READSIG: line" ) ;
   FREE ( line ) ;

   if (strlen ( error ))
      return -1 ;
   return 0 ;
}

