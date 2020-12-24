/******************************************************************************/
/*                                                                            */
/*  ARMASAVE - Save and restore ARMA models                                   */
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

static ARMA *read_arma ( char *name , FILE *fp , int *errnum ) ;
static int write_arma ( ARMA *arma , FILE *fp ) ;

struct arma_model {
   char id[20]    ; // Identification string
   int version    ; // Version number of this file (currently 1)
   int nin        ; // Number of inputs
   int nma        ; // Number of MA terms
   int nout       ; // Number of outputs
   int maxlag     ; // Maximum input lag (MA ignored)
   int nw         ; // Number of weights (length of next 2 arrays, nin+nma)
   int fixed      ; // Are output constant offsets fixed at zero?
   int nvars      ; // Total weights = nw + (fixed == 0)
   int nio        ; // Length of ins_outs
   double error   ; // Error variance (all outputs pooled)
   int h2len      ; // Length of header 2, currently 0
   char dummy[48] ; // unused data, reserved for the future, saved as 0
   } ;


/*
--------------------------------------------------------------------------------

   ARMA_SAVE
     Returns:
         0 - Normal
         1 - Unable to open file
         2 - Unable to write file

--------------------------------------------------------------------------------
*/

int arma_save ( ARMA *arma , char *filename )
{
   FILE *fp ;

   if ((fp = fopen ( filename , "wb" )) == NULL)
      return 1 ;

   if (write_arma ( arma , fp )  ||  ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }

   fclose ( fp ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   ARMA_RESTORE
      Returns (in errnum):
         0 - Normal
         1 - Cannot open file
         2 - Cannot read file
         3 - Illegal file header
         4 - Insufficient memory

--------------------------------------------------------------------------------
*/

ARMA *arma_restore ( char *armaname , char *filename , int *errnum )
{
   FILE *fp ;
   ARMA *arma ;

   if ((fp = fopen ( filename , "rb" )) == NULL) {
      *errnum = 1 ;
      return NULL ;
      }

   arma = read_arma ( armaname , fp , errnum ) ;
   if (arma == NULL) {
      fclose ( fp ) ;
      return NULL ;
      }

   fclose ( fp ) ;

   if (*errnum) {  // If file read error or insufficient memory
      MEMTEXT ( "ARMA_SAVE: file error delete ARMA" ) ;
      arma->ok = 1 ;
      delete arma ;
      arma = NULL ;
      }
   arma->trained = 1 ;          // Tell other routines it is trained

   return arma ;
}


/*
--------------------------------------------------------------------------------

   write_arma - Local routine writes all information

--------------------------------------------------------------------------------
*/

static int write_arma ( ARMA *arma , FILE *fp )
{
   int i ;
   struct arma_model h1 ;

   strcpy ( h1.id , "NPREDICT ARMA FILE " ) ;
   h1.version = 1 ;
   h1.nin = arma->nin ;
   h1.nma = arma->nma ;
   h1.nout = arma->nout ;
   h1.maxlag = arma->maxlag ;
   h1.nw = arma->nw ;
   h1.fixed = arma->fixed ;
   h1.nvars = arma->nvars ;
   h1.nio = arma->nio ;
   h1.error = arma->error ;
   
   h1.h2len = 0 ;
   for (i=0 ; i<48 ; i++)
      (h1.dummy)[i] = (char) 0 ;

   if (! fwrite ( &h1 , sizeof(h1) , 1 , fp ))
      return 1 ;

   if (! fwrite ( arma->ins_outs , h1.nio * sizeof(InputOutput) , 1 , fp ))
      return 1 ;

   if (! fwrite ( arma->var , h1.nout * sizeof(double) , 1 , fp ))
      return 1 ;

   if (! fwrite ( arma->wts , h1.nout * h1.nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   read_arma - Local routine reads all information

--------------------------------------------------------------------------------
*/

static ARMA *read_arma ( char *armaname , FILE *fp , int *errnum )
{
   int i, j ;
   struct arma_model h1 ;
   InputOutput **inout ;
   ARMA *arma ;

   if (! fread ( &h1 , sizeof(h1) , 1 , fp )) {
      *errnum = 2 ;
      return NULL ;
      }

   if (strcmp ( h1.id , "NPREDICT ARMA FILE " )) {
      *errnum = 3 ;
      return NULL ;
      }

   MEMTEXT ( "ARMASAVE: InputOutput array for restore" ) ;
   inout = (InputOutput **) MALLOC ( h1.nio * sizeof(InputOutput *) ) ;
   if (inout == NULL) {
      *errnum = 4 ;
      return NULL ;
      }

   for (i=0 ; i<h1.nio ; i++) {
      inout[i] = (InputOutput *) MALLOC ( sizeof(InputOutput) ) ;
      if (inout[i] == NULL) {
         for (j=0 ; j<i ; j++)
            FREE ( inout[j] ) ;
         FREE ( inout ) ;
         *errnum = 4 ;
         return NULL ;
         }
      if (! fread ( inout[i] , sizeof(InputOutput) , 1 , fp )) {
         for (j=0 ; j<i ; j++)
            FREE ( inout[j] ) ;
         FREE ( inout ) ;
         *errnum = 2 ;
         return NULL ;
         }
      }

   MEMTEXT ( "ARMASAVE: new ARMA for restore" ) ;
   arma = new ARMA ( armaname , h1.nin , h1.nma , h1.nout , h1.nio , inout ,
                     h1.fixed ) ;

   MEMTEXT ( "ARMASAVE: InputOutput array for restore" ) ;
   for (i=0 ; i<h1.nio ; i++)
      FREE ( inout[i] ) ;
   FREE ( inout ) ;

   if ((arma == NULL)  ||  (! arma->ok)) {  // Malloc failure?
      if (arma != NULL)
         delete arma ;
      *errnum = 4 ;
      return NULL ;
      }

   arma->nio = h1.nio ;
   arma->error = h1.error ;

   if (! fread ( arma->var , h1.nout * sizeof(double) , 1 , fp )) {
      *errnum = 2 ;
      delete arma ;
      return NULL ;
      }

   if (! fread ( arma->wts , h1.nout * h1.nvars * sizeof(double) , 1 , fp )) {
      *errnum = 2 ;
      delete arma ;
      return NULL ;
      }

   arma->trained = 1 ;   // It is trained

   *errnum = 0 ;
   return arma ;
}
