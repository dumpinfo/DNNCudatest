/******************************************************************************/
/*                                                                            */
/*  DISPLAY - Display a named signal                                          */
/*            This calls routines in GRAPHICS.CPP to achieve independence.    */
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
#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

void display ( Signal *sig , MiscParams *misc , int id )
{
   int i, j, n, row, col, ncols, rstart, rstop, cstart, cstop, istart, istop ;
   int drows, dcols, crows, ccols, xnticks, ynticks, prevcol, wide ;
   int xndigits, yndigits, xnfrac, ynfrac, r0, r1, c0, c1 ;
   double *signal, sigmin, sigmax, aspect, xmin, xmax, xdif, ymin, ymax, ydif ;
   double leftx, rightx, xscale, yscale, xtickscale, ytickscale, corrlim, val ;
   double x, xfac ;
   char msg[256] ;

   save_screen () ;

   if (sig->type == CorrelationSignal) {
      istart = 0 ;
      istop = sig->n - 1 ;
      }
   else {
      istart = misc->display_domain0 ;
      istop = misc->display_domain1 ;
      if (sig->n-1 < istop)
         istop = sig->n-1 ;
      }
   n = istop - istart + 1 ;

   signal = sig->sig ;

   if (misc->display_range < 2) {         // Optimal or Symmetric
   	sigmin = sigmax = signal[istart] ;  // Find signal limits
      for (i=istart ; i<=istop ; i++) {
         val = signal[i] ;
         if (! ((val > -1.e40)  &&  (val < 1.e40))) {
            MEMTEXT ( "DISPLAY ERROR: bad signal" ) ;
            restore_screen () ;
            printf ( "\nInternal error: Signal[%d]=%le", i, val ) ;
            return ;
            }
         if (val < sigmin)
            sigmin = val ;
         if (val > sigmax)
            sigmax = val ;
         }
      if (id == ID_PRED_DISPLAY_CONFIDENCE) {
         for (i=sig->known_n ; i<sig->known_n+sig->npred ; i++) { // Predictions
            if (i < istart)
               continue ;
            if (i > istop)
               break ;
            j = i - sig->known_n ;                       // This interval
            if (sig->mult_conf)
               val = signal[i] * sig->intervals[2*j] ;   // Lower
            else 
               val = signal[i] + sig->intervals[2*j] ;   // Lower
            if (val < sigmin)
               sigmin = val ;
            if (sig->mult_conf)
               val = signal[i] * sig->intervals[2*j+1] ; // Upper
            else 
               val = signal[i] + sig->intervals[2*j+1] ; // Upper
            if (val > sigmax)
               sigmax = val ;
            }
         }

      if ((sigmax - sigmin) < 1.e-20) {  // Center for visual effect only
         sigmin = 0.5 * (sigmin + sigmax) - .000095 ;
         sigmax = 0.5 * (sigmin + sigmax) + .000095 ;
         }

      if (sig->type == CorrelationSignal) {
         corrlim = 1.96 / sqrt ( (double) sig->source_n ) ;  // Confidence limit
         if (corrlim > sigmax)
            sigmax = corrlim ;
         if (-corrlim < sigmin)
            sigmin = -corrlim ;
         }

      if (sig->type == SpectrumDevSignal) {
         corrlim = 1.22 / sqrt ((double) sig->source_n / 2 - 1) ;  // Confidence
         if (corrlim > sigmax)
            sigmax = corrlim ;
         if (-corrlim < sigmin)
            sigmin = -corrlim ;
         }

      if (misc->display_range == 1) { // Symmetric
         if (fabs(sigmin) > fabs(sigmax)) {
            sigmax = fabs(sigmin) ;
            sigmin = -fabs(sigmin) ;
            }
         else {
            sigmin = -fabs(sigmax) ;
            sigmax = fabs(sigmax) ;
            }
         }
      } // If OPTIMAL or SYMMETRIC

   else {
      sigmin = misc->display_min ;
      sigmax = misc->display_max ;
      if (sig->type == CorrelationSignal)
         corrlim = 1.96 / sqrt ( (double) sig->source_n ) ;  // Confidence limit
      if (sig->type == SpectrumDevSignal)
         corrlim = 1.22 / sqrt ((double) sig->source_n / 2 - 1) ;  // Confidence
      }

   goto_graphics ( &drows , &dcols , &crows , &ccols , &aspect ) ;

   rstart = 3 * crows ;  // Title, ticks, tick labels
   rstop = drows - crows/2 - 2 ;

   cstart = 0 ;
   cstop = dcols - 1 ;

   write_graphics_text ( 0 , (dcols - ccols * strlen ( sig->name )) / 2 ,
                         sig->name , 15 ) ;

/*
   Compute 'pretty' tick locations.  Graphlab may have to change our specified
   number of ticks a little to keep it pretty.
*/

   if (sig->type == DataSignal) {
      leftx = misc->display_origin + istart / misc->display_rate ;
      rightx = misc->display_origin + istop / misc->display_rate ;
      }
   else if ((sig->type == SpectrumSignal) || (sig->type == SpectrumDevSignal)) {
      leftx = 0.5 * misc->display_rate * istart / sig->n ;
      rightx = 0.5 * misc->display_rate * istop / (sig->n - 1) ;
      }
   else if (sig->type == CorrelationSignal) {
      leftx = 0 ;
      rightx = n + 1 ;
      }
   best_graphlab ( leftx , rightx , 5 , 8 , &xmin , &xmax , &xdif ,
              &xndigits , &xnfrac ) ;
   xnticks = 1 + (int) ((xmax-xmin) / xdif + .1) ;

   best_graphlab ( sigmin , sigmax , 5 , 10 , &ymin , &ymax , &ydif ,
              &yndigits , &ynfrac ) ;
   ynticks = 1 + (int) ((ymax-ymin) / ydif + 0.1) ; // .1 in case slight error

/*
   Adjust graph size to make room for tick labels
*/

   i = yndigits ;               /* Space at left for Y axis tick labels */
   if (xndigits / 2 + 1  >  i)  /* And for centered X axis tick labels */
      i = xndigits / 2 + 1 ;
   cstart += i * ccols + 1 ;
   cstop -= (xndigits/2+1) * ccols + 1 ;
   ncols = cstop - cstart + 1 ;

/*
   Compute scales for data and ticks
*/

   xscale = (double) (cstop - cstart) / (xmax - xmin) ;
   xfac = (rightx - leftx) / (n-1) ;
   yscale = (double) (rstop - rstart) / (ymax - ymin) ;

   xtickscale = (double) (cstop-cstart) / (double) (xnticks-1) ;
   ytickscale = (double) (rstop-rstart) / (double) (ynticks-1) ;

/*
   Outline the graph, draw ticks, and write their labels
*/

   drawline ( cstart , rstart , cstop , rstart , 15 ) ;  /* Outline */
   drawline ( cstart , rstart , cstart , rstop , 15 ) ;
   drawline ( cstart , rstop , cstop , rstop , 15 ) ;
   drawline ( cstop , rstart , cstop , rstop , 15 ) ;
   for (i=0 ; i<ynticks ; i++) {                 /* Y ticks */
      row = rstart + i * ytickscale + 0.5 ;
      if (i  &&  (i < ynticks-1)) /* Horizontal interior lines */
         drawline ( cstart , row , cstop , row , 7 ) ;
      sprintf ( msg , "%*.*lf", yndigits, ynfrac, ymin + i * ydif ) ;
      write_graphics_text ( row - crows/2 , 0 , msg , 15 ) ;
      }

   prevcol = 0 ;
   for (i=0 ; i<xnticks ; i++) {                         /* X ticks */
      col = i * xtickscale + 0.5 ;
      if (col >= ncols)
         col = ncols-1 ;
      col += cstart ;
      drawline ( col , rstart , col , rstart - crows / 2 , 15 ) ;
      if (i)   /* Make an additional, unlabeled tick between main ones */
         drawline ( (col + prevcol) / 2  , rstart ,
                    (col + prevcol) / 2 , rstart - crows / 4, 15 ) ;
      prevcol = col ;
      sprintf ( msg , "%*.*lf", xndigits, xnfrac, xmin + i * xdif ) ;
      while (*msg == ' ')          // Remove leading blanks
         strcpy ( msg , msg+1 ) ;
      write_graphics_text ( rstart - 2 * crows ,
                            col - ccols * strlen(msg)/2 , msg , 15 ) ;
      }

/*
   Draw the graph
*/

   if ((sig->type == DataSignal)  ||  (sig->type == SpectrumSignal)
    || (sig->type == SpectrumDevSignal)) {
      if (id == ID_PRED_DISPLAY_CONFIDENCE) { // Draw confidence before signal
         for (i=sig->known_n ; i<sig->known_n+sig->npred ; i++) {
            if (i <= istart)
               continue ;
            if (i > istop)
               break ;
            x = (i - istart - 1) * xfac + leftx ;
            c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
            x = (i - istart) * xfac + leftx ;
            c1 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
            j = i - sig->known_n ;
            if (j) {
               if (sig->mult_conf)
                  val = signal[i-1] * sig->intervals[2*(j-1)] ;
               else 
                  val = signal[i-1] + sig->intervals[2*(j-1)] ;
               }
            else 
               val = signal[i-1] ;
            r0 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            if (r0 < rstart)
               r0 = rstart ;
            if (r0 > rstop)
               r0 = rstop ;
            if (sig->mult_conf)
               val = signal[i] * sig->intervals[2*j] ;
            else 
               val = signal[i] + sig->intervals[2*j] ;
            r1 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            if (r1 < rstart)
               r1 = rstart ;
            if (r1 > rstop)
               r1 = rstop ;
            drawline ( c0 , r0 , c1 , r1 , 7 ) ;

            if (j) {
               if (sig->mult_conf)
                  val = signal[i-1] * sig->intervals[2*(j-1)+1] ;
               else 
                  val = signal[i-1] + sig->intervals[2*(j-1)+1] ;
               }
            else 
               val = signal[i-1] ;
            r0 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            if (r0 < rstart)
               r0 = rstart ;
            if (r0 > rstop)
               r0 = rstop ;
            if (sig->mult_conf)
               val = signal[i] * sig->intervals[2*j+1] ;
            else 
               val = signal[i] + sig->intervals[2*j+1] ;
            r1 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            if (r1 < rstart)
               r1 = rstart ;
            if (r1 > rstop)
               r1 = rstop ;
            drawline ( c0 , r0 , c1 , r1 , 7 ) ;
            }
         } // If DISPLAY CONFIDENCE
      if (sig->type == SpectrumDevSignal) {
         row = rstart + (int) ((corrlim - ymin) * yscale + 0.5) ; // Confidence
         if ((row >= rstart)  &&  (row <= rstop))
            drawline ( cstart , row , cstop , row , 7 ) ;
         row = rstart + (int) ((-corrlim - ymin) * yscale + 0.5) ; // Confidence
         if ((row >= rstart)  &&  (row <= rstop))
            drawline ( cstart , row , cstop , row , 7 ) ;
         }
      for (i=istart+1 ; i<=istop ; i++) {  // Draw the signal
         r0 = rstart + (int) ((signal[i-1] - ymin) * yscale + 0.5) ;
         if (r0 < rstart)
            r0 = rstart ;
         if (r0 > rstop)
            r0 = rstop ;
         x = (i - istart - 1) * xfac + leftx ;
         c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
         r1 = rstart + (int) ((signal[i] - ymin) * yscale + 0.5) ;
         if (r1 < rstart)
            r1 = rstart ;
         if (r1 > rstop)
            r1 = rstop ;
         x = (i - istart) * xfac + leftx ;
         c1 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
         drawline ( c0 , r0 , c1 , r1 , 15 ) ;
         }
      } // If DataSignal or SpectrumSignal

   else if (sig->type == CorrelationSignal) {
      row = rstart + (int) ((corrlim - ymin) * yscale + 0.5) ; // Confidence
      if ((row >= rstart)  &&  (row <= rstop))
         drawline ( cstart , row , cstop , row , 15 ) ;
      row = rstart + (int) ((-corrlim - ymin) * yscale + 0.5) ; // Confidence
      if ((row >= rstart)  &&  (row <= rstop))
         drawline ( cstart , row , cstop , row , 15 ) ;
      wide = (cstop - cstart) / n ;
      r0 = rstart + (int) ((0.0 - ymin) * yscale + 0.5) ;
      for (i=0 ; i<ynticks ; i++) {            // Search all y ticks
         row = rstart + i * ytickscale + 0.5 ; // Because rounding error
         if (abs ( r0 - row ) <= 1) {          // Might make 0 off by one
            r0 = row ;                         // Which looks ugly
            break ;
            }
         }
      if (r0 < rstart)
         r0 = rstart ;
      if (r0 > rstop)
         r0 = rstop ;
      for (i=0 ; i<n ; i++) {
         c0 = cstart + (int) (xscale * (i+1) + 0.5) ;
         r1 = rstart + (int) ((signal[i] - ymin) * yscale + 0.5) ;
         if (r1 < rstart)
            r1 = rstart ;
         if (r1 > rstop)
            r1 = rstop ;
         drawline ( c0 , r0 , c0 , r1 , 15 ) ;
         if (wide && (c0 > cstart))
            drawline ( c0-1 , r0 , c0-1 , r1 , 15 ) ;
         if (wide && (c0 < cstop))
            drawline ( c0+1 , r0 , c0+1 , r1 , 15 ) ;
         if ((wide > 9)  &&  (c0 > cstart+1))
            drawline ( c0-2 , r0 , c0-2 , r1 , 15 ) ;
         if ((wide > 9)  &&  (c0 < cstop-1))
            drawline ( c0+2 , r0 , c0+2 , r1 , 15 ) ;
         }
      }

   getch () ;
   exit_graphics () ;
   restore_screen () ;
}

/*
   Windows uses this (called from signal destructor) to remove MDI child
   if it is displayed.
*/

void remove_display ( Signal * )
{
   return ;
}
