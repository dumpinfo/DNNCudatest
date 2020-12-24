/******************************************************************************/
/*                                                                            */
/*  SIG_SAVE - Save a signal to a file                                        */
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

/*
--------------------------------------------------------------------------------

   sig_save
     Returns:
         0 - Normal
         1 - Unable to open file
         2 - Unable to write file

--------------------------------------------------------------------------------
*/

int sig_save ( Signal *sig , char *filename )
{
   int i ;
   FILE *fp ;

   if ((fp = fopen ( filename , "wt" )) == NULL)
      return 1 ;

   for (i=0 ; i<sig->n ; i++) {
      if (i)
         fprintf ( fp , "\n" ) ;
      fprintf ( fp , "%.7lf" , sig->sig[i] ) ;
      }

   if (ferror ( fp )) {
      fclose ( fp ) ;
      return 2 ;
      }

   fclose ( fp ) ;
   return 0 ;
}
