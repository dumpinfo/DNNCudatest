/******************************************************************************/
/*                                                                            */
/*  SIGNAL - Implement Signal class                                           */
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

   This constructor does some slightly unusual memory stuff.
   Memory is allocated for the name, and signame is copied there.
   However, memory is NOT allocated for the signal.  The pointer 'signal'
   is simply copied to the member 'sig'.
   The destructor frees BOTH of these blocks.

--------------------------------------------------------------------------------
*/

Signal::Signal ( char *signame , int length , double *signal )
{
   char msg[256] ;

   type = DataSignal ;
   sig = signal ;
   n = known_n = length ;
   centered = detrended = offsetted = scaled = standardized = 0 ;
   cent = pivot = slope = offst = mean = 0.0 ;
   scal = std = 1.0 ;
   differenced = 0 ;
   seasonal = 0 ;
   source_n = 0 ;
   sleads = NULL ;
   npred = 0 ;
   intervals = NULL ;
   mult_conf = 0 ;

   sprintf ( msg , "Signal constructor (%s)", signame ) ;
   MEMTEXT ( msg ) ;
   name = (char *) MALLOC ( strlen ( signame ) + 1 ) ;
   if (! name) {
      n = 0 ;
      FREE ( sig ) ;
      return ;
      }
   strcpy ( name , signame ) ;
}

Signal::~Signal ()
{
   if (! n)
      return ;
   remove_display ( this ) ; // Windows must remove MDI child if displayed
   MEMTEXT ( "Signal destructor (2)" ) ;
   FREE ( name ) ;
   FREE ( sig ) ;
   if (seasonal) {
      MEMTEXT ( "Signal destructor: sleads" ) ;
      FREE ( sleads ) ;
      }
   if (intervals != NULL) {
      FREE ( intervals ) ;
      intervals = NULL ;
      }
}

/*
--------------------------------------------------------------------------------

   Replace a signal's data vector only

--------------------------------------------------------------------------------
*/

void Signal::replace ( int nnew , int n_protect , double *x )
{
   if (! nnew)  // Should never happen
      return ;

   if (n_protect > n)
      n_protect = n ;

   if (n_protect > nnew)
      n_protect = nnew ;

   if (n_protect) {
      memcpy ( x , sig , n_protect * sizeof(double) ) ;
      known_n = n_protect ;
      }
   else 
      known_n = nnew ;

   if (n) {     // Should always be true
      remove_display ( this ) ; // Windows must remove MDI child if displayed
      MEMTEXT ( "SIGNAL: replace" ) ;
      FREE ( sig ) ;
      }

   type = DataSignal ;
   n = nnew ;
   sig = x ;
}

/*
--------------------------------------------------------------------------------

   Transformations

--------------------------------------------------------------------------------
*/

void Signal::center ()
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   cent = 0.0 ;
   for (i=0 ; i<n ; i++)
      cent += sig[i] ;
   cent /= n ;

   for (i=0 ; i<n ; i++)
      sig[i] -= cent ;

   centered = 1 ;
   type = DataSignal ;
}

int Signal::median_center ()
{
   int i ;
   double *work ;

   MEMTEXT ( "SIGNAL: Alloc and free work" ) ;
   work = (double *) MALLOC ( n * sizeof(double) ) ;
   if (work == NULL)
      return 1 ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   memcpy ( work , sig , n * sizeof(double) ) ;
   qsort ( 0 , n-1 , work ) ;
   cent = work[n/2] ;

   for (i=0 ; i<n ; i++)
      sig[i] -= cent ;

   centered = 1 ;
   type = DataSignal ;
   FREE ( work ) ;
   return 0 ;
}

void Signal::undo_center ( Signal *sigptr )
{
   int i ;
   double move ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->centered)
         move = sigptr->cent ;
      else
         return ;
      }
   else
      move = cent ;

   for (i=0 ; i<n ; i++)
      sig[i] += move ;
   type = DataSignal ;
}

void Signal::detrend ()
{
   int i ;
   double xmean, ymean, x, y, xy, xx ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   xmean = 0.5 * (n - 1) ;

   ymean = xy = xx = 0.0 ;
   for (i=0 ; i<n ; i++)
      ymean += sig[i] ;
   ymean /= n ;

   for (i=0 ; i<n ; i++) {
      x = i - xmean ;
      y = sig[i] - ymean ;
      xx += x * x ;
      xy += x * y ;
      }

   slope = xy / xx ;
   pivot = xmean ;

   for (i=0 ; i<n ; i++)
      sig[i] -= (i - xmean) * slope ;

   detrended = 1 ;
   type = DataSignal ;
}

void Signal::undo_detrend ( Signal *sigptr )
{
   int i ;
   double xmean, rate ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->detrended) {
         rate = sigptr->slope ;
         xmean = sigptr->pivot ;
         }
      else
         return ;
      }
   else {
      rate = slope ;
      xmean = pivot ;
      }

   for (i=0 ; i<n ; i++)
      sig[i] += (i - xmean) * rate ;
   type = DataSignal ;
}

void Signal::offset ( double val )
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   offst = val ;

   for (i=0 ; i<n ; i++)
      sig[i] += offst ;

   offsetted = 1 ;
   type = DataSignal ;
}

void Signal::undo_offset ( Signal *sigptr )
{
   int i ;
   double move ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->offsetted)
         move = sigptr->offst ;
      else
         return ;
      }
   else
      move = offst ;

   for (i=0 ; i<n ; i++)
      sig[i] -= move ;
   type = DataSignal ;
}

void Signal::scale ( double val )
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   scal = val ;

   for (i=0 ; i<n ; i++)
      sig[i] *= scal ;

   scaled = 1 ;
   type = DataSignal ;
}

void Signal::undo_scale ( Signal *sigptr )
{
   int i ;
   double move ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->scaled)
         move = sigptr->scal ;
      else
         return ;
      }
   else
      move = scal ;

   if (move == 0.0)
      return ;

   for (i=0 ; i<n ; i++)
      sig[i] /= move ;
   type = DataSignal ;
}

void Signal::standardize ()
{
   int i ;
   double diff ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   mean = std = 0.0 ;
   for (i=0 ; i<n ; i++)
      mean += sig[i] ;
   mean /= n ;

   for (i=0 ; i<n ; i++) {
      diff = sig[i] - mean ;
      std += diff * diff ;
      }
   std = sqrt ( std / n ) ;

   for (i=0 ; i<n ; i++)
      sig[i] = (sig[i] - mean) / std ;

   standardized = 1 ;
   type = DataSignal ;
}

void Signal::undo_standardize ( Signal *sigptr )
{
   int i ;
   double xbar, xstd ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->standardized) {
         xbar = sigptr->mean ;
         xstd = sigptr->std ;
         }
      else
         return ;
      }
   else {
      xbar = mean ;
      xstd = std ;
      }

   for (i=0 ; i<n ; i++)
      sig[i] = xstd * sig[i] + xbar ;
   type = DataSignal ;
}

void Signal::difference ( int degree )
{
   int i, pass ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   for (pass=0 ; pass<degree ; pass++) {
      --n ;
      leads[pass] = sig[0] ;   // Preserve first point for undo
      for (i=0 ; i<n ; i++)
         sig[i] = sig[i+1] - sig[i] ;
      }

   differenced = degree ;

   if (intervals != NULL) {
      MEMTEXT ( "SIGNAL: difference freeing intervals" ) ;
      FREE ( intervals ) ;
      intervals = NULL ;
      npred = 0 ;
      }

   type = DataSignal ;
   known_n -= degree ;
}

int Signal::undo_difference ( Signal *sigptr )
{
   int i, pass, passes ;
   double temp, sum, *dptr ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->differenced)
         passes = sigptr->differenced ;
      else
         return 0 ;
      }
   else
      passes = differenced ;

   MEMTEXT ( "SIGNAL: undo_difference" ) ;
   dptr = (double *) REALLOC ( sig , (n+passes) * sizeof(double) ) ;
   if (dptr == NULL)
      return 1 ;
   sig = dptr ;
   
   for (pass=0 ; pass<passes ; pass++) {
      if (sigptr != NULL)
         sum = sigptr->leads[passes-pass-1] ;
      else
         sum = leads[passes-pass-1] ;
      for (i=0 ; i<n ; i++) {
         temp = sig[i] ;
         sig[i] = sum ;
         sum += temp ;
         }
      sig[n++] = sum ;
      }
   type = DataSignal ;
   known_n += passes ;
   return 0 ;
}

int Signal::seasonal_diff ( int period )
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   MEMTEXT ( "SIGNAL: seasonal sleads" ) ;
   sleads = (double *) MALLOC ( period * sizeof(double) ) ;
   if (sleads == NULL)
      return 1 ;

   for (i=0 ; i<period ; i++)
      sleads[i] = sig[i] ;

   n -= period ;
   for (i=0 ; i<n ; i++)
      sig[i] = sig[i+period] - sig[i] ;

   seasonal = period ;
   type = DataSignal ;

   if (intervals != NULL) {
      MEMTEXT ( "SIGNAL: seasonal_diff freeing intervals" ) ;
      FREE ( intervals ) ;
      intervals = NULL ;
      npred = 0 ;
      }
   known_n -= period ;
   return 0 ;
}

int Signal::undo_seasonal_diff ( Signal *sigptr )
{
   int p, i, per ;
   double temp, sum, *dptr ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   if (sigptr != NULL) {
      if (sigptr->seasonal)
         per = sigptr->seasonal ;
      else
         return 0 ;
      }
   else
      per = seasonal ;

   MEMTEXT ( "SIGNAL: undo_seasonal" ) ;
   dptr = (double *) REALLOC ( sig , (n+per) * sizeof(double) ) ;
   if (dptr == NULL)
      return 1 ;
   sig = dptr ;

   for (p=0 ; p<per ; p++) {
      if (sigptr != NULL)
         sum = sigptr->sleads[p] ;
      else
         sum = sleads[p] ;
      for (i=p ; i<n ; i+=per ) {
         temp = sig[i] ;
         sig[i] = sum ;
         sum += temp ;
         }
      sig[i] = sum ;
      }
   n += per ;
   known_n += per ;
   type = DataSignal ;
   return 0 ;
}

void Signal::integrate ( int period )
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   for (i=period ; i<n ; i++ )
      sig[i] += sig[i-period] ;

   type = DataSignal ;
}

void Signal::siglog ()
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   for (i=0 ; i<n ; i++ ) {
      if (sig[i] > 0.0)
         sig[i] = log ( sig[i] ) ;
      else 
         sig[i] = -1.e20 ;
      }

   type = DataSignal ;
}

void Signal::sigexp ()
{
   int i ;

   remove_display ( this ) ; // Windows must remove MDI child if displayed

   for (i=0 ; i<n ; i++ )
      sig[i] = exp ( sig[i] ) ;

   type = DataSignal ;
}
