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
/*  AN1 - Learn by anneal method 1                                            */
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

void LayerNet::an1 ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int itry, user_quit ;
   long seed ;
   double best_err ;
   char msg[80] ;
   LayerNet *worknet, *bestnet ;

/*
   Allocate scratch memory
*/

   MEMTEXT ( "AN1::learn new worknet, bestnet" ) ;
   worknet = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                            nout , 0 , 0 ) ;
   bestnet = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                            nout , 0 , 1 ) ;

   if ((worknet == NULL)  ||  (! worknet->ok)
    || (bestnet == NULL)  ||  (! bestnet->ok)) {
      memory_message ( "to learn" ) ;
      if (worknet != NULL)
         delete worknet ;
      if (bestnet != NULL)
         delete bestnet ;
      errtype = 0 ;
      return ;
      }

   best_err = 1.e30 ;
   for (itry=1 ; itry<=lptr->retries+1 ; itry++) {

      user_quit = anneal1 ( tptr , lptr , worknet , 1 , itry ) ;
      if (neterr < best_err) {
         best_err = neterr ;
         copy_weights ( bestnet , this ) ;
         }

      sprintf ( msg , "Try %d  err=%lf  best=%lf", itry, neterr, best_err ) ;
      normal_message ( msg ) ;

      if (user_quit  ||  (neterr < lptr->quit_err))
         break ;

      seed = flrand() - (long) (itry * 97) ;   // Insure new seed for anneal
      sflrand ( seed ) ;
      zero_weights () ;  // Retry random
      }

   copy_weights ( this , bestnet ) ;
   neterr = best_err ;
   MEMTEXT ( "AN1::learn delete worknet, bestnet" ) ;
   delete worknet ;
   delete bestnet ;
   return ;
}

