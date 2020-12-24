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
/*  AN2 - Learn by anneal method 2                                            */
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

void LayerNet::an2 ( TrainingSet *tptr , struct LearnParams *lptr )
{
   int itry, user_quit ;
   long seed ;
   double best_err ;
   char msg[80] ;
   LayerNet *worknet1, *worknet2, *bestnet ;

/*
   Allocate scratch memory
*/

   MEMTEXT ( "AN2::learn new worknet1&2, bestnet" ) ;
   worknet1 = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                             nout , 0 , 0 ) ;
   worknet2 = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                             nout , 0 , 0 ) ;
   bestnet = new LayerNet ( model , outmod , outlin , nin , nhid1 , nhid2 ,
                            nout , 0 , 1 ) ;

   if ((worknet1 == NULL)  ||  (! worknet1->ok)
    || (worknet2 == NULL)  ||  (! worknet2->ok)
    || (bestnet == NULL)   ||  (! bestnet->ok)) {
      memory_message ( "to learn" ) ;
      if (worknet1 != NULL)
         delete worknet1 ;
      if (worknet2 != NULL)
         delete worknet2 ;
      if (bestnet != NULL)
         delete bestnet ;
      errtype = 0 ;
      return ;
      }

   best_err = 1.e30 ;
   for (itry=1 ; itry<=lptr->retries+1 ; itry++) {

      user_quit = anneal2 ( tptr , lptr , worknet1 , worknet2 , 1 ) ;
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
   MEMTEXT ( "AN2::learn delete worknet1&2, bestnet" ) ;
   delete worknet1 ;
   delete worknet2 ;
   delete bestnet ;
   return ;
}


