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
/*  WT_SAVE - Save and restore learned weights to/from disk files             */
/*                                                                            */
/*  This returns 1 if error, else 0.                                          */
/*                                                                            */
/*  It handles common errors, opens and closes files, etc.                    */
/*  The actual work is done in the appropriate network virtual function.      */
/*                                                                            */
/*  The 'updating' parameter would be input 1 if this routine is called for   */
/*  automatic updating during training, else 0.                               */
/*                                                                            */
/*  The file header is as follows:                                            */
/*                                                                            */
/*    Header 1:                                                               */
/*      PNN NETWORK FILE - ASCII string for identification purposes           */
/*      version - Version number of this file (currently 1)                   */
/*      model - NETMOD_BASIC et cetera                                        */
/*      kernel - KERNEL_GAUSS et cetera                                       */
/*      maxmom - Maximum moment for GCNN                                      */
/*      n_in - Number of input neurons (sum of lengths)                       */
/*      n_out - Number of output neurons                                      */
/*      outmodel - AUTO / CLASSIF / GENERAL                                   */
/*      h2len - Length of header 2, currently 0                               */
/*      48 bytes of unused data, reserved for the future                      */
/*                                                                            */
/*    Header 2:  currently not used                                           */
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

static int write_header ( PNNet *net , FILE *fp ) ;
static PNNet *read_header ( FILE *fp ) ;

struct wt_header1 {
   char id[20]    ; // Identification string
   int version    ; // Version number of this file (currently 1)
   int model      ; // NETMOD_BASIC etc. 
   int kernel     ; // KERNEL_GAUSS etc. 
   int maxmom     ; // GCNN max moment
   int n_in       ; // Number of input neurons (sum of lengths)
   int n_out      ; // Number of output neurons
   int outmodel   ; // AUTO / CLASSIF / GENERAL
   int h2len      ; // Length of header 2, currently 0
   char dummy[48] ; // unused data, reserved for the future, saved as 0
   } ;

/*
--------------------------------------------------------------------------------

   WT_SAVE

--------------------------------------------------------------------------------
*/

void wt_save ( PNNet *net , int updating , char *name )
{
   char msg[81] ;
   FILE *fp ;

   if ((fp = fopen ( name , "wb" )) == NULL) {
      if (! updating) {
         strcpy ( msg , "Cannot open WEIGHT file " ) ;
         strcat ( msg , name ) ;
         error_message ( msg ) ;
         }
      return ;
      }

   if (write_header ( net , fp )  ||  net->wt_save ( fp ) ||
       ferror ( fp )) {
      if (! updating) {
         strcpy ( msg , "Could not write WEIGHT file " ) ;
         strcat ( msg , name ) ;
         error_message ( msg ) ;
         }
      fclose ( fp ) ;
      return ;
      }

   fclose ( fp ) ;
}


/*
--------------------------------------------------------------------------------

   WT_RESTORE

--------------------------------------------------------------------------------
*/

PNNet *wt_restore ( char *name )
{
   char msg[81] ;
   FILE *fp ;
   PNNet *net ;

   if ((fp = fopen ( name , "rb" )) == NULL) {
      strcpy ( msg , "Cannot open WEIGHT file " ) ;
      strcat ( msg , name ) ;
      error_message ( msg ) ;
      return NULL ;
      }

   net = read_header ( fp ) ;
   if (net == NULL) {
      strcpy ( msg , "Could not read WEIGHT file " ) ;
      strcat ( msg , name ) ;
      error_message ( msg ) ;
      fclose ( fp ) ;
      return NULL ;
      }

   net->wt_restore ( fp ) ;
   fclose ( fp ) ;

   if (! net->ok) {  // Set to 0 if file read error or insufficient memory
      strcpy ( msg , "Could not read NETWORK file " ) ;
      strcat ( msg , name ) ;
      error_message ( msg ) ;
      MEMTEXT ( "WT_SAVE: file error delete network" ) ;
      net->ok = 1 ;  // Tells destructor to free vectors
      delete net ;
      net = NULL ;
      }
   net->errtype = 1 ;          // Tell other routines net is trained
   return net ;
}


/*
--------------------------------------------------------------------------------

   write_header - Local routine writes weight file header

--------------------------------------------------------------------------------
*/

static int write_header ( PNNet *net , FILE *fp )
{
   int i ;
   struct wt_header1 h1 ;

   strcpy ( h1.id , "PNN NETWORK FILE" ) ;
   h1.version = 1 ;
   h1.model = net->model ;
   h1.kernel = net->kernel ;
   h1.maxmom = net->maxmom ;
   h1.n_in = net->nin ;
   h1.n_out = net->nout ;
   h1.outmodel = net->outmod ;
   h1.h2len = 0 ;
   for (i=0 ; i<48 ; i++)
      (h1.dummy)[i] = (char) 0 ;

   if (! fwrite ( &h1 , sizeof(h1) , 1 , fp ))
      return 1 ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   read_header - Local routine reads network file header

--------------------------------------------------------------------------------
*/

static PNNet *read_header ( FILE *fp )
{
   int model, nin, nout, outmod, kernl, mom ;
   struct wt_header1 h1 ;
   PNNet *net ;

   if (! fread ( &h1 , sizeof(h1) , 1 , fp ))
      return NULL ;

   if (strcmp ( h1.id , "PNN NETWORK FILE" )) {
      error_message ( "This is not a PNN NETWORK file." ) ;
      return NULL ;
      }

   model = h1.model ;
   kernl = h1.kernel ;
   mom = h1.maxmom ;
   nin = h1.n_in ;
   nout = h1.n_out ;
   outmod = h1.outmodel ;

   MEMTEXT ( "WT_SAVE: new network for restore" ) ;
   if (model == NETMOD_BASIC)
      net = new PNNbasic ( kernl , outmod , nin , nout ) ;
   else if (model == NETMOD_SEPVAR)
      net = new PNNsepvar ( kernl , outmod , nin , nout ) ;
   else if (model == NETMOD_SEPCLASS)
      net = new PNNsepclass ( kernl , outmod , nin , nout ) ;
   else if (model == NETMOD_GCNN)
      net = new GCNN ( mom , 0 , outmod , nin , nout ) ;
   else if (model == NETMOD_GCNN_EW)
      net = new GCNN ( mom , 1 , outmod , nin , nout ) ;

   if ((net == NULL)  ||  (! net->ok)) {  // Malloc failure?
      memory_message ( "to create network." ) ;
      if (net != NULL)
         delete net ;
      return NULL ;
      }

   return net ;
}
