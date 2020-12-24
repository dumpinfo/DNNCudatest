/******************************************************************************/
/*                                                                            */
/*  ORTHSAVE - Save and restore ORTHOGONALIZATION models                      */
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
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

static Orthog *read_orth ( char *name , FILE *fp , int *errnum ) ;
static int write_orth ( Orthog *orth , FILE *fp ) ;

struct orth_model {
   char id[24]    ; // Identification string
   int version    ; // Version number of this file (currently 1)
   int nin        ; // Number of inputs
   int nfacs      ; // Number of output factors
   int type       ; // Orthogonalization type (1-3)
   int h2len      ; // Length of header 2, currently 0
   char dummy[48] ; // unused data, reserved for the future, saved as 0
   } ;


/*
--------------------------------------------------------------------------------

   Orth_save
     Returns:
         0 - Normal
         1 - Unable to open file
         2 - Unable to write file

--------------------------------------------------------------------------------
*/

int orth_save ( Orthog *orth , char *filename )
{
   FILE *fp ;

   if ((fp = fopen ( filename , "wb" )) == NULL)
      return 1 ;

   if (write_orth ( orth , fp )  ||  ferror ( fp )) { // Writes header
      fclose ( fp ) ;
      return 2 ;
      }

   if (orth->type == 3) {
      if (orth->discrim->save ( fp )  ||  ferror ( fp )) {
         fclose ( fp ) ;
         return 2 ;
         }
      }
   else {
      if (orth->princo->save ( fp )  ||  ferror ( fp )) {
         fclose ( fp ) ;
         return 2 ;
         }
      }

   fclose ( fp ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   Orthog_RESTORE
      Returns (in errnum):
         0 - Normal
         1 - Cannot open file
         2 - Cannot read file
         3 - Illegal file header
         4 - Insufficient memory

--------------------------------------------------------------------------------
*/

Orthog *orth_restore ( char *orthname , char *filename , int *errnum )
{
   FILE *fp ;
   Orthog *orth ;
   PrincoData *princo ;
   Discrim *discrim ;

   if ((fp = fopen ( filename , "rb" )) == NULL) {
      *errnum = 1 ;
      return NULL ;
      }

   orth = read_orth ( orthname , fp , errnum ) ;
   if (orth == NULL) {
      fclose ( fp ) ;
      return NULL ;
      }

   *errnum = 4 ;    // Be a pessimist and wait for constructor to change it

   if (orth->type == 3) {
      discrim = new Discrim ( fp , errnum ) ;
      if (*errnum == 0)
         orth->discrim = discrim ;
      else if (discrim != NULL)
         delete discrim ;
      }
   else {
      princo = new PrincoData ( fp , errnum ) ;
      if (*errnum == 0)
         orth->princo = princo ;
      else if (princo != NULL)
         delete princo ;
      }

   fclose ( fp ) ;

   if (*errnum) {  // If file read error or insufficient memory
      MEMTEXT ( "Orthog_SAVE: error delete Orthog" ) ;
      delete orth ;
      orth = NULL ;
      }

   return orth ;
}


/*
--------------------------------------------------------------------------------

   write_orth - Local routine writes all information

--------------------------------------------------------------------------------
*/

static int write_orth ( Orthog *orth , FILE *fp )
{
   int i ;
   struct orth_model h1 ;

   strcpy ( h1.id , "NPREDICT Orthog FILE   " ) ;
   h1.version = 1 ;
   h1.nin = orth->nin ;
   h1.nfacs = orth->nfacs ;
   h1.type = orth->type ;
   
   h1.h2len = 0 ;
   for (i=0 ; i<48 ; i++)
      (h1.dummy)[i] = (char) 0 ;

   if (! fwrite ( &h1 , sizeof(h1) , 1 , fp ))
      return 1 ;

   if (! fwrite ( orth->lags , h1.nin * sizeof(unsigned) , 1 , fp ))
      return 1 ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   read_orth - Local routine reads all information

--------------------------------------------------------------------------------
*/

static Orthog *read_orth ( char *orthname , FILE *fp , int *errnum )
{
   unsigned *lags ;
   struct orth_model h1 ;
   Orthog *orth ;

   if (! fread ( &h1 , sizeof(h1) , 1 , fp )) {
      *errnum = 2 ;
      return NULL ;
      }

   if (strcmp ( h1.id , "NPREDICT Orthog FILE   " )) {
      *errnum = 3 ;
      return NULL ;
      }

   MEMTEXT ( "ORTHSAVE: lags array for restore" ) ;
   lags = (unsigned *) MALLOC ( h1.nin * sizeof(unsigned) ) ;
   if (lags == NULL) {
      *errnum = 4 ;
      return NULL ;
      }

   if (! fread ( lags , h1.nin * sizeof(unsigned) , 1 , fp )) {
      *errnum = 2 ;
      FREE ( lags ) ;
      return NULL ;
      }

   MEMTEXT ( "OrthogSAVE: new Orthog for restore" ) ;
   orth = new Orthog ( orthname , h1.nin , h1.nfacs , h1.type , lags ) ;

   MEMTEXT ( "ORTHSAVE: lags array for restore" ) ;
   FREE ( lags ) ;

   if ((orth == NULL)  ||  (! orth->ok)) {  // Malloc failure?
      if (orth != NULL)
         delete orth ;
      *errnum = 4 ;
      return NULL ;
      }

   *errnum = 0 ;
   return orth ;
}
