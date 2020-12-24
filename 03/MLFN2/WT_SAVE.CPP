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
/*      MLFN WEIGHT FILE - ASCII string for identification purposes           */
/*      version - Version number of this file (currently 1)                   */
/*      model - NETMOD_REAL et cetera                                         */
/*      linear - Is output activation linear?                                 */
/*      n_in - Number of input neurons (sum of lengths)                       */
/*      n_hid1 - Number of hidden layer one neruons                           */
/*      n_hid2 - Ditto for hidden layer two                                   */
/*      n_out - Number of output neurons                                      */
/*      outmodel - AUTO / CLASSIF / MAPPING                                   */
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

static int write_header ( LayerNet *net , FILE *fp ) ;
static LayerNet *read_header ( FILE *fp ) ;

struct wt_header1 {
   char id[20]    ; // Identification string
   int version    ; // Version number of this file (currently 1)
   int model      ; // NETMOD_REAL, NETMOD_COMPLEX etc. 
   int linear     ; // Is output activation linear?
   int n_in       ; // Number of input neurons (sum of lengths)
   int n_hid1     ; // Number of hidden layer one neruons
   int n_hid2     ; // Ditto for hidden layer two
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

void wt_save ( LayerNet *net , int updating , char *name )
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

   if (write_header ( net , fp ) || net->wt_save ( fp ) ||
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

LayerNet *wt_restore ( char *name )
{
   char msg[81] ;
   FILE *fp ;
   LayerNet *net ;

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

   if (! net->ok) {  // Set to 0 if file read error
      strcpy ( msg , "Could not read WEIGHT file " ) ;
      strcat ( msg , name ) ;
      error_message ( msg ) ;
      MEMTEXT ( "WT_SAVE: file error delete network" ) ;
      net->ok = 1 ;  // Tells destructor to free vectors
      delete net ;
      net = NULL ;
      }
   return net ;
}


/*
--------------------------------------------------------------------------------

   write_header - Local routine writes weight file header

--------------------------------------------------------------------------------
*/

static int write_header ( LayerNet *net , FILE *fp )
{
   int i ;
   struct wt_header1 h1 ;

   strcpy ( h1.id , "MLFN WEIGHT FILE" ) ;
   h1.version = 1 ;
   h1.model = net->model ;
   h1.linear = net->outlin ;
   h1.n_in = net->nin ;
   h1.n_hid1 = net->nhid1 ;
   h1.n_hid2 = net->nhid2 ;
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

   read_header - Local routine reads weight file header

--------------------------------------------------------------------------------
*/

static LayerNet *read_header ( FILE *fp )
{
   int model, lin, nin, nh1, nh2, nout, outmod ;
   struct wt_header1 h1 ;
   LayerNet *net ;

   if (! fread ( &h1 , sizeof(h1) , 1 , fp ))
      return NULL ;

   if (strcmp ( h1.id , "MLFN WEIGHT FILE" )) {
      error_message ( "This is not a MLFN WEIGHT file." ) ;
      return NULL ;
      }

   model = h1.model ;
   lin = h1.linear ;
   nin = h1.n_in ;
   nh1 = h1.n_hid1 ;
   nh2 = h1.n_hid2 ;
   nout = h1.n_out ;
   outmod = h1.outmodel ;

   MEMTEXT ( "WT_SAVE: new network for restore" ) ;
   net = new LayerNet ( model , outmod , lin , nin , nh1 , nh2 , nout , 1 , 0 );

   if ((net == NULL)  ||  (! net->ok)) {  // Malloc failure?
      memory_message ( "to create network." ) ;
      if (net != NULL)
         delete net ;
      return NULL ;
      }

   return net ;
}
