/******************************************************************************/
/*                                                                            */
/*  NETWORK - Virtual base class for all neural networks                      */
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


/*
--------------------------------------------------------------------------------

   Constructor and destructor (for Network, the base class)

   Failure (due to insufficient memory) is indicated in two ways.
   The 'ok' flag will be set to zero to indicate to child classes that
   they should abort and to the caller that there was failure.
   Also, 'out' will be set to NULL.  In case this constructor succeeds but
   the child fails, a real possibility, the child will set 'ok' to zero
   but 'out' here will be valid.  That tells this base class destructor to
   free its memory.

--------------------------------------------------------------------------------
*/

Network::Network ( char *netname , NetParams *net_params )
{
   int i, j ;

   MEMTEXT ( "Network constructor" ) ;

   netparams = *net_params ;  // This is sometimes handy to have around
                              // Especially when creating a scratch network

   if (netname)
      strcpy ( name , netname ) ;
   else 
      name[0] = 0 ;

   n_inputs = net_params->n_inputs ;
   n_outputs = net_params->n_outputs ;
   model = net_params->net_model ;
   output_mode = net_params->out_model ;

   out = NULL ;
   leads = NULL ;
   errtype = 0 ;  // Flag that not yet trained

   ok = 0 ;   // Indicates failure of malloc (What a pessimist!)

   out = (double *) MALLOC ( n_outputs * sizeof(double) ) ;
   lags = (unsigned *) MALLOC ( n_inputs * sizeof(unsigned) ) ;
   if ((out == NULL)  ||  (lags == NULL)) {
      if (out != NULL) {
         FREE ( out ) ;
         out = NULL ;
         }
      if (lags != NULL) {
         FREE ( lags ) ;
         lags = NULL ;
         }
      return ;
      }

   if (output_mode == OUTMOD_CLASSIFICATION) { // One for each class, plus reject
      classnames = (char **) MALLOC ( n_outputs * sizeof(char *) ) ;
      if (classnames == NULL) {
         FREE ( out ) ;
         out = NULL ;
         FREE ( lags ) ;
         lags = NULL ;
         return ;
         }
      for (i=0 ; i<n_outputs ; i++) {
         classnames[i] = (char *) MALLOC ( strlen(net_params->classnames[i])+1);
         if (classnames[i] == NULL) {
            for (j=0 ; j<i ; j++)
               FREE ( classnames[j] ) ;
            FREE ( classnames ) ;
            classnames = NULL ;
            FREE ( out ) ;
            out = NULL ;
            FREE ( lags ) ;
            lags = NULL ;
            return ;
            }
         strcpy ( classnames[i] , net_params->classnames[i] ) ;
         }
      }
   else {
      classnames = NULL ;
      leads = (unsigned *) MALLOC ( n_outputs * sizeof(unsigned)) ;
      if (leads == NULL) {
         FREE ( out ) ;
         out = NULL ;
         FREE ( lags ) ;
         lags = NULL ;
         return ;
         }
      }

   ok = 1 ;            // Indicate to caller that all mallocs succeeded
}

Network::~Network ()
{
   int i ;

   if (! ok)      // If constructor's mallocs failed
      return ;    // there is nothing to free

   MEMTEXT ( "Network destructor" ) ;
   FREE ( out ) ;
   FREE ( lags ) ;
   if (leads != NULL)
      FREE ( leads ) ;
   if (classnames != NULL) {
      for (i=0 ; i<n_outputs ; i++)
         FREE ( classnames[i] ) ;
      FREE ( classnames ) ;
      }
}
