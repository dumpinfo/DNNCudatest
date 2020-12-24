/******************************************************************************/
/*                                                                            */
/*  IN_OUT - Keep current INPUT/OUTPUT definitions                            */
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

void clear_io ( int is_input , int *nio , InputOutput ***inputs_outputs )
{
   int i, n ;
   InputOutput **ioptr ;

   if (! *nio)
      return ;

   MEMTEXT ( "IN_OUT: clear_io compressing" ) ;
   ioptr = *inputs_outputs ;              // Array of pointers to InputOutput
   n = 0 ;                                // This many will remain
   for (i=0 ; i<*nio ; i++) {             // Check all that exist right now
      if (ioptr[i]->is_input == is_input) // If this is one to clear
         FREE ( ioptr[i] ) ;              // Then free it
      else                                // If we keep this one
         ioptr[n++] = ioptr[i] ;          // Copy it to beginning of array
      }

   MEMTEXT ( "IN_OUT: adjust array" ) ;
   if (n)                                 // If any are left
      *inputs_outputs = (InputOutput **) REALLOC (
                                *inputs_outputs , n * sizeof(InputOutput *) ) ;
   else {
      FREE ( *inputs_outputs ) ;
      *inputs_outputs = NULL ;
      }

   *nio = n ;
}


int append_io ( int is_input , char *rest , 
                 int n_signals , Signal **signals ,
                 int *nio , InputOutput ***inputs_outputs )
{
   int i, j, nin, nout, n, which, minlag, maxlag ;
   char name[256], *nameptr ;
   InputOutput **ioptr ;

/*
   Find this name in the signal array
*/

   while (*rest == ' ')   // Bypass leading blanks
      ++rest ;

   nameptr = &name[0] ;
   while (*rest  &&  (*rest != ' '))
      *nameptr++ = *rest++ ;
   *nameptr = 0 ;

   for (which=0 ; which<n_signals ; which++) {
      if (! strcmp ( name , signals[which]->name))
         break ;
      }
   if (which == n_signals)
      return 1 ;

/*
   Parse signal lags(s)
*/

   while (*rest == ' ')   // Bypass leading blanks
      ++rest ;

   if ((*rest >= '0')  &&  (*rest <= '9')) {
      minlag = atoi ( rest ) ;
      while ((*rest >= '0')  &&  (*rest <= '9'))
         ++rest ;
      while ((*rest == ' ')  ||  (*rest == '-'))
         ++rest ;
      if ((*rest >= '0')  &&  (*rest <= '9'))
         maxlag = atoi ( rest ) ;
      else
         maxlag = minlag ;
      }
   else
      minlag = maxlag = 0 ;

   if (minlag > maxlag)
      return 3 ;

/*
   Add this to the InputOutput array
*/

   MEMTEXT ( "IN_OUT: append_io" ) ;
   if (*nio)
      ioptr = (InputOutput **) REALLOC (
                         *inputs_outputs , ((*nio)+1) * sizeof(InputOutput *) );
   else 
      ioptr = (InputOutput **) MALLOC ( ((*nio)+1) * sizeof(InputOutput *) ) ;

   if (ioptr == NULL)
      return 2 ;

   *inputs_outputs = ioptr ;
   n = *nio ;

   ioptr[n] = (InputOutput *) MALLOC ( sizeof(InputOutput) ) ;
   if (ioptr[n] == NULL)
      return 2 ;

   ++*nio ;

   ioptr[n]->is_input = is_input ;
   ioptr[n]->which = which ;
   ioptr[n]->minlag = minlag ;
   ioptr[n]->maxlag = maxlag ;
   ioptr[n]->shock = -1 ;

/*
   Compute ordinal positions
*/

   nin = nout = 0 ;
   for (i=0 ; i<=n ; i++) {
      if (ioptr[i]->is_input) {
         for (j=0 ; j<i ; j++) {
            if (ioptr[j]->is_input  &&  (ioptr[j]->which == ioptr[i]->which))
               break ;
            }
         if (i == j)
            ioptr[i]->ordinal = nin++ ;
         else 
            ioptr[i]->ordinal = ioptr[j]->ordinal ;
         }
      else {
         for (j=0 ; j<i ; j++) {
            if ((!ioptr[j]->is_input)  &&  (ioptr[j]->which == ioptr[i]->which))
               break ;
            }
         if (i == j)
            ioptr[i]->ordinal = nout++ ;
         else 
            ioptr[i]->ordinal = ioptr[j]->ordinal ;
         }
      }

/*
   Compute is_other.  For inputs, this is the ordinal position of an existing
   output of the same signal, and -1 if none.  Converse for outputs.
*/

   for (i=0 ; i<=n ; i++) {
      for (j=0 ; j<=n ; j++) {
         if ((ioptr[j]->is_input  !=  ioptr[i]->is_input) &&
             (ioptr[j]->which  ==  ioptr[i]->which))
            break ;
         }
      if (j <= n)
         ioptr[i]->is_other = ioptr[j]->ordinal ;
      else 
         ioptr[i]->is_other = -1 ;
      }

   return 0 ;
}

