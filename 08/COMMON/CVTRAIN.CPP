/******************************************************************************/
/*                                                                            */
/*  CVTRAIN - Cross validation training                                       */
/*                                                                            */
/*  This returns -1 if insufficient memory, 1 if interrupted, else 0.         */
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

int cvtrain ( TrainingSet *tptr , Network *net , struct NetParams *net_params ,
              struct LearnParams *lptr , double *cverror )
{
   int ret, n, user_quit, exclude, ntested ;
   double *dptr0, *dptr1, err, tot_err ;
   TrainingSet *cvptr ;
   Network *cvnet ;

   *cverror = -1.0 ;  // Flag that it is totally invalid

/*
   We need a local copy of the training set
*/

   MEMTEXT ( "CVTRAIN starting" ) ;
   cvptr = new TrainingSet ( tptr->output_mode , tptr->n_inputs ,
                             tptr->n_outputs , 0 , NULL);
   if (cvptr == NULL)
      return -1 ;

   *cvptr = *tptr ;       // Invoke assignment operator to duplicate it

   if (! cvptr->ntrain) { // Insufficient memory to copy?
      delete cvptr ;
      return -1 ;         // Return error flag
      }

/*
   Start by doing a normal training operation.
   That way, if the user aborts at least they've got something.
*/

   ret = net->learn ( tptr , lptr ) ;

   if (ret) {             // If the user aborted, quit now
      delete cvptr ;
      return ret ;
      }

   if (tptr->ntrain < 2) {
      delete cvptr ;
      return 1 ;
      }

/*
   Here is the cross validation loop
*/

   user_quit = 0 ;
   tot_err = 0.0 ;
   exclude = cvptr->ntrain ;   // Will exclude this case
   ntested = 0 ;

   while (exclude--) {   // Exclude each training case

      cvptr->ntrain = tptr->ntrain - 1 ; // We are excluding one case
      if (exclude)       // Copy cases below excluded case
         memcpy ( cvptr->data , tptr->data ,
                  tptr->size * exclude * sizeof(double) ) ;
      n = cvptr->ntrain - exclude ;
      if (n) {           // Copy cases above excluded case
         dptr0 = tptr->data + tptr->size * (exclude + 1) ;
         dptr1 = cvptr->data + tptr->size * exclude ;
         memcpy ( dptr1 , dptr0 , tptr->size * n * sizeof(double) ) ;
         }

      MEMTEXT ( "CVTRAIN: new Network" ) ;
      if (net_params->net_model == NETMOD_PNN)
         cvnet = new PNNbasic ( "" , net_params ) ;
      else if (net_params->net_model == NETMOD_SEPVAR)
         cvnet = new PNNsepvar ( "" , net_params ) ;
      else if (net_params->net_model == NETMOD_SEPCLASS)
         cvnet = new PNNsepclass ( "" , net_params ) ;
      else if (net_params->net_model == NETMOD_MLFN)
         cvnet = new MLFN ( "" , net_params ) ;
      if ((cvnet == NULL)  ||  (! cvnet->ok)) {  // Malloc failure?
         if (cvnet != NULL)
            delete cvnet ;
         delete cvptr ;
         return -1 ;
         }

      ret = cvnet->learn ( cvptr , lptr ) ;
      if (ret) {             // If the user aborted, quit now
         MEMTEXT ( "CVTRAIN: abort deleting cvnet" ) ;
         delete cvnet ;
         user_quit = ret ;
         break ;
         }

      cvptr->ntrain = 1 ;  // Make a test set having just the excluded case
      memcpy ( cvptr->data , tptr->data + exclude * tptr->size ,
               tptr->size * sizeof(double) ) ;

      err = cvnet->trial_error ( cvptr ) ;

      if (err < 0.0) {
         user_quit = 1 ;
         MEMTEXT ( "CVTRAIN: abort deleting cvnet" ) ;
         delete cvnet ;
         break ;
         }

      tot_err += err ;
      ++ntested ;

      MEMTEXT ( "CVTRAIN: delete Network" ) ;
      delete cvnet ;
      } // for all excluded

   MEMTEXT ( "CVTRAIN: deleting cvptr" ) ;
   delete cvptr ;

   if (ntested)
      *cverror = tot_err / ntested ;

   return user_quit ;
}
