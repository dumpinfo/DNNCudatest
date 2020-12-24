/******************************************************************************/
/*                                                                            */
/*  GRAPHLAB - Nice graph labels.  From Glassner, "Graphics Gems" pg 657.     */
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

/*
   Compute 'nice' labels for a graph.  From Glassner, "Graphics Gems" pg 657.
*/

#include <stdlib.h>
#include <math.h>

extern double safe_power ( double val , int n ) ;
extern double pretty ( double x , int round ) ;

void graphlab ( 
   double dmin ,   // Minimum data value
   double dmax ,   // and maximum
   int nticks ,    // Approximate (!) number of ticks to use
   double *gmin ,  // Minimum value actually plotted
   double *gmax ,  // (And max) for a 'nice' plot range
   double *dif ,   // Increment between ticks
   int *ntot ,     // Total digits for labels (includes dec pt & sign)
   int *nfrac      // Number of fractional digits to show
   )
{
   double range, big ;

   range = pretty ( dmax - dmin , 0 ) ;
   *dif = pretty ( range / (nticks-1) , 1 ) ;
   *gmax = ceil  ( dmax / *dif )  *  *dif ;
   *gmin = floor ( dmin / *dif )  *  *dif ;
   *nfrac = -floor ( log10 (*dif) ) ;
   if (*nfrac < 0)
      *nfrac = 0 ;
   *ntot = *nfrac + 2 ;    /* Leading digit + decimal point */
   if (*gmin < 0.)
      ++*ntot ;            /* Minus sign */
   big = abs (*gmin)  ;
   if (big < abs (*gmax) )
      big = abs (*gmax) ;
   if (big >= 10.)
      *ntot += floor ( log10 ( big ) ) ;  /* Extra digits */
}

static double safe_power ( double val , int n )
{
   double prod ;

   prod = 1.0 ;

   if (n>0) {
      while (n--)
         prod *= val ;
      }
   else {
      while (n++)
         prod /= val ;
      }
   return prod ;
}

static double pretty ( double x , int round )
{
   int exponent ;
   double factor, test ;

   exponent = floor ( log10 ( x ) ) ;
   test = x  /  safe_power ( 10.0 , exponent ) ;

   if (round) {
      if (test < 1.5)
         factor = 1.0 ;
      else if (test < 3.0)
         factor = 2.0 ;
      else if (test < 7.0)
         factor = 5.0 ;
      else
         factor = 10.0 ;
      }
   else {
      if (test <= 1.0)
         factor = 1.0 ;
      else if (test <= 2.0)
         factor = 2.0 ;
      else if (test <= 5.0)
         factor = 5.0 ;
      else
         factor = 10.0 ;
      }

   return factor * safe_power ( 10.0 , exponent ) ;
}

/*
--------------------------------------------------------------------------------

   This is a modified version that finds the narrowest range among a range
   of nticks.

--------------------------------------------------------------------------------
*/

void best_graphlab ( 
   double dmin ,   // Minimum data value
   double dmax ,   // and maximum
   int min_ticks , // Minimum number of ticks to use
   int max_ticks , // Maximum number of ticks to use
   double *gmin ,  // Minimum value actually plotted
   double *gmax ,  // (And max) for a 'nice' plot range
   double *dif ,   // Increment between ticks
   int *ntot ,     // Total digits for labels (includes dec pt & sign)
   int *nfrac      // Number of fractional digits to show
   )
{
   int i, ibest ;
   double best_range = 1.e40 ;

   ibest = min_ticks ;
   for (i=min_ticks ; i<=max_ticks ; i++) {
      graphlab ( dmin , dmax , i , gmin , gmax , dif , ntot , nfrac ) ;
      if ((*gmax - *gmin) <= best_range) {
         best_range = *gmax - *gmin ;
         ibest = i ;
         }
      }

   graphlab ( dmin , dmax , ibest , gmin , gmax , dif , ntot , nfrac ) ;
}
