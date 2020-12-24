/******************************************************************************/
/*                                                                            */
/*  WT_SAVE - Save and restore neural network weights.                        */
/*                                                                            */
/*  This relies on the virtual member functions wt_save and wt_restore.       */
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

static Network *read_header ( char *netname , FILE *fp , int *errnum ) ;
static int write_header ( Network *net , FILE *fp ) ;

struct npred_header {
   char id[24]    ; // Identification string
   int f_ver      ; // Version number of this file (currently 1)
   int model      ; // NETMOD_? in CONST.H
   int n_inputs   ; // Number of inputs
   int n_outputs  ; // Number of outputs
   int output_mode; // OUTMOD_? in CONST.H
   // PNN family parameters
   int kernel     ; // KERNEL_GAUSS etc.
   // MLFN family parameters
   int domain     ; // DOMAIN_?
   int linear     ; // Is output activation linear?
   int n_hidden1  ; // Hidden-layer-one neuron count
   int n_hidden2  ; // Hidden-layer-two neuron count
   int h2_length  ; // Header 2 length, 0 in this version
   char dummy[64] ; // Reserved area for future expansion, guaranteed 0
   } ;


/*
--------------------------------------------------------------------------------

   WT_SAVE
     Returns:
         0 - Normal
         1 - Unable to open file
         2 - Unable to write file

--------------------------------------------------------------------------------
*/

int wt_save ( Network *net , char *filename )
{
   FILE *fp ;

   if ((fp = fopen ( filename , "wb" )) == NULL)
      return 1 ;

   if (write_header ( net , fp )  ||  net->wt_save ( fp )  ||  ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }

   fclose ( fp ) ;
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   WT_RESTORE
      Returns:
         0 - Normal
         1 - Cannot open file
         2 - Cannot read file
         3 - Illegal file header
         4 - Insufficient memory

--------------------------------------------------------------------------------
*/

Network *wt_restore ( char *netname , char *filename , int *errnum )
{
   FILE *fp ;
   Network *net ;

   if ((fp = fopen ( filename , "rb" )) == NULL) {
      *errnum = 1 ;
      return NULL ;
      }

   net = read_header ( netname , fp , errnum ) ;
   if (net == NULL) {
      fclose ( fp ) ;
      return NULL ;
      }

   *errnum = net->wt_restore ( fp ) ;
   fclose ( fp ) ;

   if (*errnum) {  // If file read error or insufficient memory
      MEMTEXT ( "WT_SAVE: file error delete network" ) ;
      net->ok = 1 ;  // Destructor must free memory
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

static int write_header ( Network *net , FILE *fp )
{
   int i, n ;
   struct npred_header header ;

   strcpy ( header.id , "NPREDICT NETWORK FILE" ) ;
   header.f_ver = 1 ;
   header.model = net->model ;
   header.n_inputs = net->n_inputs ;
   header.n_outputs = net->n_outputs ;
   header.output_mode = net->output_mode ;
   if ((net->model == NETMOD_PNN)  ||
       (net->model == NETMOD_SEPVAR) ||
       (net->model == NETMOD_SEPCLASS))
      header.kernel = ((PNNet *) net) -> kernel ;
   if (net->model == NETMOD_MLFN) {
      header.domain = ((MLFN *) net) -> domain ;
      header.linear = ((MLFN *) net) -> outlin ;
      header.n_hidden1 = ((MLFN *) net) -> nhid1 ;
      header.n_hidden2 = ((MLFN *) net) -> nhid2 ;
      }
   header.h2_length = 0 ;
   for (i=0 ; i<64 ; i++)
      (header.dummy)[i] = (char) 0 ;

   if (! fwrite ( &header , sizeof(header) , 1 , fp ))
      return 1 ;

   if (net->output_mode == OUTMOD_CLASSIFICATION) {
      for (i=0 ; i<net->n_outputs ; i++) {
         n = strlen ( net->classnames[i] ) + 1 ;
         if (! fwrite ( &n , sizeof(int) , 1, fp))
            return 1 ;
         if (! fwrite ( net->classnames[i] , n , 1 , fp ))
            return 1 ;
         }
      }

   if ((net->model == NETMOD_MLFN) && (((MLFN *)net)->domain != DOMAIN_REAL))
      n = net->n_inputs * 2 ;
   else 
      n = net->n_inputs ;
   if (! fwrite ( net->lags , n * sizeof(unsigned) , 1 , fp ))
      return 1 ;

   if (net->output_mode == OUTMOD_MAPPING) {
      if ((net->model==NETMOD_MLFN) && (((MLFN *)net)->domain==DOMAIN_COMPLEX))
         n = net->n_outputs * 2 ;
      else 
         n = net->n_outputs ;
      if (! fwrite ( net->leads , n * sizeof(unsigned) , 1 , fp ))
         return 1 ;
      }

   return 0 ;
}



/*
--------------------------------------------------------------------------------

   read_header - Local routine reads network file header

--------------------------------------------------------------------------------
*/

static Network *read_header ( char *netname , FILE *fp , int *errnum )
{
   int i, j, n ;
   char **names ;
   struct npred_header header ;
   Network *net ;
   NetParams net_params ;

   if (! fread ( &header , sizeof(header) , 1 , fp )) {
      *errnum = 2 ;
      return NULL ;
      }

   if (strcmp ( header.id , "NPREDICT NETWORK FILE" )) {
      *errnum = 3 ;
      return NULL ;
      }

   if (header.output_mode == OUTMOD_CLASSIFICATION) {
      MEMTEXT ( "WT_SAVE: read_header classnames" ) ;
      names = (char **) MALLOC ( header.n_outputs * sizeof(char *) ) ;
      if (names == NULL) {
         *errnum = 4 ;
         return NULL ;
         }
      for (i=0 ; i<header.n_outputs ; i++) {
         if (! fread ( &n , sizeof(int) , 1, fp)) {
            *errnum = 2 ;
            return NULL ;
            }
         names[i] = (char *) MALLOC ( n ) ;
         if (names[i] == NULL) {
            for (j=0 ; j<i ; j++)
               FREE ( names[j] ) ;
            FREE ( names ) ;
            *errnum = 4 ;
            return NULL ;
            }
         if (! fread ( names[i] , n , 1 , fp )) {
            *errnum = 2 ;
            return NULL ;
            }
         }
      net_params.classnames = names ;
      }
   else 
      net_params.classnames = NULL ;

   net_params.net_model = header.model ;
   net_params.n_inputs = header.n_inputs ;
	net_params.n_outputs = header.n_outputs ;
   net_params.out_model = header.output_mode ;
   net_params.kernel = header.kernel ;
   net_params.domain = header.domain ;
   net_params.linear = header.linear ;
	net_params.n_hidden1 = header.n_hidden1 ;
	net_params.n_hidden2 = header.n_hidden2 ;

   MEMTEXT ( "WT_SAVE: new network for restore" ) ;
   if (header.model == NETMOD_PNN)
      net = new PNNbasic ( netname , &net_params ) ;
   else if (header.model == NETMOD_SEPVAR)
      net = new PNNsepvar ( netname , &net_params ) ;
   else if (header.model == NETMOD_SEPCLASS)
      net = new PNNsepclass ( netname , &net_params ) ;
   else if (header.model == NETMOD_MLFN) {
      if (header.domain != DOMAIN_REAL)
         net_params.n_inputs *= 2 ;
      if (header.domain == DOMAIN_COMPLEX)
         net_params.n_outputs *= 2 ;
      net = new MLFN ( netname , &net_params ) ;
      }

   if (header.output_mode == OUTMOD_CLASSIFICATION) {
      MEMTEXT ( "WT_SAVE: read_header classnames" ) ;
      for (i=0 ; i<header.n_outputs ; i++)
         FREE ( names[i] ) ;
      FREE ( names ) ;
      }

   if ((net == NULL)  ||  (! net->ok)) {  // Malloc failure?
      if (net != NULL)
         delete net ;
      *errnum = 4 ;
      return NULL ;
      }

   if ((net->model == NETMOD_MLFN)  &&  (((MLFN*)net)->domain != DOMAIN_REAL))
      n = header.n_inputs * 2 ;
   else 
      n = header.n_inputs ;
   if (! fread ( net->lags , n * sizeof(unsigned) , 1 , fp )) {
      *errnum = 2 ;
      return NULL ;
      }

   if (net->output_mode == OUTMOD_MAPPING) {
      if ((net->model == NETMOD_MLFN) && (((MLFN*)net)->domain==DOMAIN_COMPLEX))
         n = header.n_outputs * 2 ;
      else 
         n = header.n_outputs ;
      if (! fread ( net->leads , n * sizeof(unsigned) , 1 , fp )) {
         *errnum = 2 ;
         return NULL ;
         }
      }

   *errnum = 0 ;
   return net ;
}
