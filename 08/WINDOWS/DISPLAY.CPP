/******************************************************************************/
/*                                                                            */
/*  DISPLAY - Display a named signal                                          */
/*                                                                            */
/*  There are two routines here.                                              */
/*                                                                            */
/*  'display' is called from PROCESS.CPP in response to ID_PRED_DISPLAY.      */
/*  It precomputes as much as possible from the signal and stores this        */
/*  information in a static area here.                                        */
/*  Then it does the Windows busywork of creating a new MDI child window.     */
/*                                                                            */
/*  'Paint' is the overridden virtual routine called in response to WM_PAINT. */
/*  It actually graphs the signal.                                            */
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

#include <owl\owlpch.h>
#include <\owl\listbox.h>
#include <\owl\button.h>
#include <\owl\static.h>
#include <\owl\validate.h>
#include <\owl\groupbox.h>
#include <owl\inputdia.h>
#include <owl\dialog.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#pragma hdrstop

#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

#include "prdcmdi1.h"           // MDI child declarations
#include "prdcmdic.h"           // MDI client declarations
#include "extern.h"             // Externals for Windows stuff

/*
--------------------------------------------------------------------------------

   We keep an array of all open signal display windows.
   The signal destructor calls 'remove_display' in this module.
   That routine does nothing in the DOS version.  But in Windows, the
   display will still be up.  So remove_display scans this array of
   signals displayed, finds this one, and deletes it.

   Also, the child destructor calls 'purge_display' which removes that
   display from the list.

--------------------------------------------------------------------------------
*/

#define DISPLAY_MAX_WINDOWS 256

   static int n_displayed = 0 ;
	static npredictMDIChild *children[DISPLAY_MAX_WINDOWS] ;


/*
--------------------------------------------------------------------------------

   display - Called from PROCESS to initialize and create MDI child window

--------------------------------------------------------------------------------
*/

void display ( Signal *sig , MiscParams *misc , int id )
{
	int i, j, n ;
	double *signal, sigmin, sigmax, cl, val ;
	npredictMDIChild *child ;

/*
   Search the display list for a signal having this name.
   If there, just use that existing child.  Else create the MDI child window.
*/

   for (i=0 ; i<n_displayed ; i++) {
      child = children[i] ;
      if (! strcmp ( child->signal->name , sig->name ))
         break ;
      }

   if (i == n_displayed) {
      child = new npredictMDIChild ( mdiClient , sig->name ) ;
      MEMTEXT ( "--->DISPLAY made new child" ) ;
      child->background_color = RGB ( 255 , 255 , 255 ) ;
      child->Create();
      children[n_displayed++] = child ;
      }
   else
      child->Invalidate();

	child->signal = sig ;
   child->command_id = id ;

   if (sig->type == CorrelationSignal) {
      child->istart = 0 ;
      child->istop = sig->n - 1 ;
      }
   else {
      child->istart = misc->display_domain0 ;
      child->istop = misc->display_domain1 ;
      if (sig->n-1 < child->istop)
         child->istop = sig->n-1 ;
      }

   n = child->istop - child->istart + 1 ;

   signal = sig->sig ;

   if (misc->display_range < 2) {                // Optimal or Symmetric
   	sigmin = sigmax = signal[child->istart] ;  // Find signal limits
      for (i=child->istart ; i<=child->istop ; i++) {
         val = signal[i] ;
         if (val < sigmin)
            sigmin = val ;
         if (val > sigmax)
            sigmax = val ;
         }
      if (id == ID_PRED_DISPLAY_CONFIDENCE) {
         for (i=sig->known_n ; i<sig->known_n+sig->npred ; i++) { // Predictions
            if (i < child->istart)
               continue ;
            if (i > child->istop)
               break ;
            j = i - sig->known_n ;                    // This interval
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
         cl = 1.96 / sqrt ( (double) sig->source_n ) ;  // Confidence limits
         if (cl > sigmax)
            sigmax = cl ;
         if (-cl < sigmin)
            sigmin = -cl ;
         child->corrlim = cl ;
         }

      if (sig->type == SpectrumDevSignal) {
         cl = 1.22 / sqrt ((double) sig->source_n / 2 - 1) ;  // Confidence
         if (cl > sigmax)
            sigmax = cl ;
         if (-cl < sigmin)
            sigmin = -cl ;
         child->corrlim = cl ;
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
         } // If SYMMETRIC
      } // If OPTIMAL or SYMMETRIC

   else {
      sigmin = misc->display_min ;
      sigmax = misc->display_max ;
      if (sig->type == CorrelationSignal)
         child->corrlim = 1.96 / sqrt ( (double) sig->source_n ) ; // Confidence
      if (sig->type == SpectrumDevSignal)
         child->corrlim = 1.22 / sqrt ((double) sig->source_n / 2 - 1) ;
      }

/*
	Compute 'pretty' tick locations.  Graphlab may have to change our specified
	number of ticks a little to keep it pretty.
*/

   if (sig->type == DataSignal) {
      child->leftx = misc->display_origin + child->istart / misc->display_rate ;
      child->rightx = misc->display_origin + child->istop / misc->display_rate ;
      }
   else if ((sig->type == SpectrumSignal) || (sig->type == SpectrumDevSignal)) {
      child->leftx = 0.5 * misc->display_rate * child->istart / sig->n ;
      child->rightx = 0.5 * misc->display_rate * child->istop / (sig->n - 1) ;
      }
   else if (sig->type == CorrelationSignal) {
      child->leftx = 0 ;
      child->rightx = n + 1 ;
      }
	best_graphlab ( child->leftx , child->rightx , 5 , 8 ,
		&child->xmin , &child->xmax , &child->xdif ,
		&child->xndigits , &child->xnfrac ) ;
	child->xnticks = 1 + (int) ((child->xmax - child->xmin) / child->xdif + 0.1);

	best_graphlab ( sigmin , sigmax , 5 , 10 ,
		&child->ymin , &child->ymax , &child->ydif ,
		&child->yndigits , &child->ynfrac ) ;
	child->ynticks = 1 + (int) ((child->ymax - child->ymin) / child->ydif + 0.1);
}

/*
--------------------------------------------------------------------------------

   remove_display - Remove this signal from the displayed children
                    This is called from the Signal destructor, as well as
                    from several routines in SIGNAL.CPP that modify the signal.

--------------------------------------------------------------------------------
*/

void remove_display ( Signal *sig )
{
   int i ;

   for (i=0 ; i<n_displayed ; i++) {
      if ((children[i])->signal == sig) {
         (children[i])->Destroy() ;
         MEMTEXT ( "--->DISPLAY remove_display delete child" ) ;
         delete children[i] ;
         return ;
         }
      }
}


/*
--------------------------------------------------------------------------------

   purge_display - Remove this signal from the list.
                   This is called from the display child destructor.

--------------------------------------------------------------------------------
*/

void purge_display ( npredictMDIChild *child )
{
   int i, n ;

   n = 0 ;
   for (i=0 ; i<n_displayed ; i++) {
      if (children[i] != child)
         children[n++] = children[i] ;
      }

   n_displayed = n ;
}


/*
--------------------------------------------------------------------------------

	Paint - Called in response to WM_PAINT for MDI child
	
	This uses an origin at the lower-left corner.  Thus, we must invert
	all Y coordinates before calling windows routines!

--------------------------------------------------------------------------------
*/

void npredictMDIChild::Paint ( TDC& dc , BOOL , TRect& )
{
	int i, j, n, b, row, col, rstart, rstop, cstart, cstop, prevcol ;
	int xlab_height, xlab_width, ylab_height, ylab_width, wide ;
	int xlab_lw, xlab_rw, tick_height, font_height, r0, r1, c0 ;
	double xscale, yscale, xtickscale, ytickscale, *sig, val ;
   double x, xfac ;
	char msg[256] ;
	TRect rect ;
	SIZE size ;
	TColor bkgnd, *lines ;
	TBrush *brush ;
	TPen *pen, *second_pen ;
   TFont *font ;

	n = istop - istart + 1 ;
	sig = signal->sig ;
	
	dc.SetMapMode ( MM_TEXT ) ;
	dc.SetROP2 ( R2_COPYPEN ) ;
	dc.SetBkMode ( OPAQUE ) ;
	bkgnd = dc.GetNearestColor ( TColor ( background_color ) ) ;
	dc.SetBkColor ( bkgnd ) ;
	GetClientRect ( rect ) ;
	b = rect.bottom ;   // We use origin at bottom, so b-y adapts to Windows

/*
	Paint the background
*/

	brush = new TBrush ( bkgnd ) ;
	dc.SelectObject ( *brush ) ;
	pen = new TPen ( bkgnd ) ;
	dc.SelectObject ( *pen ) ;
	dc.Rectangle ( rect ) ;
	dc.RestoreBrush () ;
	delete brush ;
	dc.RestorePen () ;
	delete pen ;
		
/*
   Setup main drawing pen and secondary pen for hash lines
*/

   lines = new TColor ( 0 , 0 , 0 ) ;
	pen = new TPen ( *lines , 1 , PS_SOLID ) ;
	second_pen = new TPen ( *lines , 1 , PS_DASH ) ;

/*
	Compute the maximum height and width of the axis tick labels.
	This lets us position the graph optimally.
*/
	
	xlab_height = xlab_width = ylab_height = ylab_width = xlab_lw = xlab_rw = 0 ;
	
	font_height = rect.bottom / 20 ;
	if (rect.right / 24 < font_height)
		font_height = rect.right / 24 ;

	if (font_height < 8)
      font_height = 0 ;

	if (font_height) {
		font = new TFont ( "Helvetica" , font_height ) ;
		dc.SelectObject ( *font ) ;
   	for (i=0 ; i<xnticks ; i++) {
   		sprintf ( msg , "%*.*lf", xndigits, xnfrac, xmin + i * xdif ) ;
   		GetTextExtentPoint ( dc , msg , strlen(msg) , &size ) ;
   		if (! i)
   			xlab_lw = size.cx ;      // Width of leftmost x label
   		if (i == xnticks-1)
   			xlab_rw = size.cx ;      // And rightmost x label
   		if (size.cx > xlab_width)
   			xlab_width = size.cx ;
   		if (size.cy > xlab_height)
   			xlab_height = size.cy ;
   		}

   	for (i=0 ; i<ynticks ; i++) {
   		sprintf ( msg , "%*.*lf", yndigits, ynfrac, ymin + i * ydif ) ;
   		GetTextExtentPoint ( dc , msg , strlen(msg) , &size ) ;
			if (size.cx > ylab_width)
				ylab_width = size.cx ;
			if (size.cy > ylab_height)
				ylab_height = size.cy ;
			}

		tick_height = xlab_height / 2 ;
		}

	else {
		tick_height = rect.bottom / 64  +  2 ;
      font = NULL ;
      }

/*
   Last check on label display.  Are X-axis labels to wide?
*/

   if ((xlab_width * xnticks) > (3 * rect.right / 4)) {
      font_height = 0 ;
	   xlab_height = xlab_width = ylab_height = ylab_width = 0 ;
      xlab_lw = xlab_rw = 0 ;
		tick_height = rect.bottom / 64  +  2 ;
      if (font != NULL) {
	      dc.RestoreFont () ;
      	delete font ;
         font = NULL ;
         }
      }

/*
	Compute the graph extent based on these label metrics
*/

	
	rstart = tick_height + xlab_height + 1  ;  // Ticks, X tick labels
	rstop = rect.bottom - ylab_height/2 - 2 ;  // Top half of topmost label

	if (xlab_lw / 2  >  ylab_width)            // If left half of X label biggest
		cstart = xlab_lw / 2 + 1 ;              // Use it for offset
	else
		cstart = ylab_width + 2 ;               // Otherwise Y labels determine it

	cstop = rect.right - xlab_rw / 2 - 2 ;     // Leave room for rightmost label
	
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

	dc.SelectObject ( *pen ) ;
	dc.MoveTo ( cstart , b-rstart ) ;
	dc.LineTo ( cstop , b-rstart ) ;
	dc.LineTo ( cstop , b-rstop ) ;
	dc.LineTo ( cstart , b-rstop ) ;
	dc.LineTo ( cstart , b-rstart ) ;
	dc.RestorePen () ;

	for (i=0 ; i<ynticks ; i++) {                 /* Y ticks */
		row = rstart + i * ytickscale + 0.5 ;
		if (i  &&  (i < ynticks-1)) {  // Horizontal interior lines
         dc.SelectObject ( *second_pen ) ;
			dc.MoveTo ( cstart , b-row ) ;
			dc.LineTo ( cstop , b-row ) ;
         dc.RestorePen () ;
			}
      if (font_height) {
   		sprintf ( msg , "%*.*lf", yndigits, ynfrac, ymin + i * ydif ) ;
   		rect.bottom = b - (row - ylab_height / 2) ;
   		rect.top = rect.bottom - ylab_height - 1 ;
   		rect.right = cstart - 1 ;
   		rect.left = rect.right - ylab_width - 1 ;
         dc.SelectObject ( *pen ) ;
   		dc.DrawText ( msg , -1 , rect , DT_SINGLELINE | DT_BOTTOM | DT_RIGHT );
         dc.RestorePen () ;
         }
		}

   dc.SelectObject ( *pen ) ;
	prevcol = 0 ;
	for (i=0 ; i<xnticks ; i++) {                         /* X ticks */
		col = cstart + i * xtickscale + 0.5 ;
		dc.MoveTo ( col , b-rstart ) ;
		dc.LineTo ( col , b-(rstart - tick_height) ) ;
		if (i) {  /* Make an additional, unlabeled tick between main ones */
			dc.MoveTo ( (col + prevcol) / 2  , b-rstart ) ;
			dc.LineTo ( (col + prevcol) / 2 , b-(rstart - tick_height/2) ) ;
			}
		prevcol = col ;
		if (font_height) {
			sprintf ( msg , "%*.*lf", xndigits, xnfrac, xmin + i * xdif ) ;
         while (*msg == ' ')          // Remove leading blanks
            strcpy ( msg , msg+1 ) ;
			rect.top = b - (rstart - tick_height) ;
			rect.bottom = rect.top + xlab_height - 1 ;
			rect.right = col + xlab_width / 2 ;
			rect.left = rect.right - xlab_width - 1 ;
			dc.DrawText ( msg , -1 , rect , DT_SINGLELINE | DT_BOTTOM | DT_CENTER);
			}
		}
   dc.RestorePen () ;

   if (font_height) {
	   dc.RestoreFont () ;
   	delete font ;
      }

/*
   Draw the graph
*/

   if ((signal->type == DataSignal)  ||  (signal->type == SpectrumSignal)
    || (signal->type == SpectrumDevSignal)) {
      if ((command_id == ID_PRED_DISPLAY_CONFIDENCE) && // Draw confidence first
          (signal->known_n <= istop)) {                 // If visible
         delete second_pen ;    // No longer needed for hash lines, so use for
         second_pen = new TPen ( *lines , 1 , PS_DOT ) ; // Confidence lines
         dc.SelectObject ( *second_pen ) ;
         i = signal->known_n - 1 ;
         if (i >= istart)
            r0 = rstart + (int) ((sig[i] - ymin) * yscale) ;
         else {
            i = istart ;
            j = i - signal->known_n ;
            if (signal->mult_conf)
               val = sig[i] * signal->intervals[2*j] ;
            else 
               val = sig[i] + signal->intervals[2*j] ;
            r0 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            }
         if (r0 < rstart)
            r0 = rstart ;
         if (r0 > rstop)
            r0 = rstop ;
         r0 = b - r0 ;
         x = (i - istart) * xfac + leftx ;
         c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
         dc.MoveTo ( c0 , r0 ) ;
         for (i=i+1 ; i<signal->known_n+signal->npred ; i++) {
            if (i > istop)
               break ;
            j = i - signal->known_n ;
            if (signal->mult_conf)
               val = sig[i] * signal->intervals[2*j] ;
            else 
               val = sig[i] + signal->intervals[2*j] ;
            r0 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            if (r0 < rstart)
               r0 = rstart ;
            if (r0 > rstop)
               r0 = rstop ;
            r0 = b - r0 ;
            x = (i - istart) * xfac + leftx ;
            c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
      		dc.LineTo ( c0 , r0 ) ;
            }

         i = signal->known_n - 1 ;
         if (i >= istart)
            r0 = rstart + (int) ((sig[i] - ymin) * yscale) ;
         else {
            i = istart ;
            j = i - signal->known_n ;
            if (signal->mult_conf)
               val = sig[i] * signal->intervals[2*j] ;
            else 
               val = sig[i] + signal->intervals[2*j] ;
            r0 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            }
         if (r0 < rstart)
            r0 = rstart ;
         if (r0 > rstop)
            r0 = rstop ;
         r0 = b - r0 ;
         x = (i - istart) * xfac + leftx ;
         c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
         dc.MoveTo ( c0 , r0 ) ;
         for (i=signal->known_n ; i<signal->known_n+signal->npred ; i++) {
            if (i > istop)
               break ;
            j = i - signal->known_n ;
            if (signal->mult_conf)
               val = sig[i] * signal->intervals[2*j+1] ;
            else 
               val = sig[i] + signal->intervals[2*j+1] ;
            r0 = rstart + (int) ((val - ymin) * yscale + 0.5) ;
            if (r0 < rstart)
               r0 = rstart ;
            if (r0 > rstop)
               r0 = rstop ;
            r0 = b - r0 ;
            x = (i - istart) * xfac + leftx ;
            c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
      		dc.LineTo ( c0 , r0 ) ;
            }
         dc.RestorePen () ;
         }

      if (signal->type == SpectrumDevSignal) {
         delete second_pen ;    // No longer needed for hash lines, so use for
         second_pen = new TPen ( *lines , 1 , PS_DOT ) ; // Confidence lines
         dc.SelectObject ( *second_pen ) ;
         row = rstart + (int) ((corrlim - ymin) * yscale + 0.5) ; // Confidence
         if ((row >= rstart)  &&  (row <= rstop)) {
            dc.MoveTo ( cstart , b-row ) ;
            dc.LineTo ( cstop , b-row ) ;
            }
         row = rstart + (int) ((-corrlim - ymin) * yscale + 0.5) ; // Confidence
         if ((row >= rstart)  &&  (row <= rstop)) {
            dc.MoveTo ( cstart , b-row ) ;
            dc.LineTo ( cstop , b-row ) ;
            }
         dc.RestorePen () ;
         }

      dc.SelectObject ( *pen ) ;
      r0 = rstart + (int) ((sig[istart] - ymin) * yscale) ;
      if (r0 < rstart)
         r0 = rstart ;
      if (r0 > rstop)
         r0 = rstop ;
      r0 = b - r0 ;
      x = leftx ;
      c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
      dc.MoveTo ( c0 , r0 ) ;
   	for (i=istart+1 ; i<=istop ; i++) {
         r0 = rstart + (int) ((sig[i] - ymin) * yscale + 0.5) ;
         if (r0 < rstart)
            r0 = rstart ;
         if (r0 > rstop)
            r0 = rstop ;
         r0 = b - r0 ;
         x = (i - istart) * xfac + leftx ;
         c0 = cstart + (int) (xscale * (x-xmin) + 0.5) ;
   		dc.LineTo ( c0 , r0 ) ;
   		}
      dc.RestorePen () ;
      } // If DataSignal or SpectrumSignal

   else if (signal->type == CorrelationSignal) {
      delete second_pen ;    // No longer needed for hash lines, so use for
      second_pen = new TPen ( *lines , 1 , PS_DOT ) ; // Confidence lines
      dc.SelectObject ( *second_pen ) ;
      row = rstart + (int) ((corrlim - ymin) * yscale + 0.5) ; // Confidence
      if ((row >= rstart)  &&  (row <= rstop)) {
         dc.MoveTo ( cstart , b-row ) ;
         dc.LineTo ( cstop , b-row ) ;
         }
      row = rstart + (int) ((-corrlim - ymin) * yscale + 0.5) ; // Confidence
      if ((row >= rstart)  &&  (row <= rstop)) {
         dc.MoveTo ( cstart , b-row ) ;
         dc.LineTo ( cstop , b-row ) ;
         }
      dc.RestorePen () ;
      wide = (cstop - cstart) / n ;
      r0 = rstart + (int) ((0.0 - ymin) * yscale + 0.5) ;
      for (i=0 ; i<ynticks ; i++) {            // Search all horizontal ticks
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
      r0 = b - r0 ;
      dc.SelectObject ( *pen ) ;
   	for (i=0 ; i<n ; i++) {
         c0 = cstart + (int) (xscale * (i+1) + 0.5) ;
         r1 = rstart + (int) ((sig[i] - ymin) * yscale + 0.5) ;
         if (r1 < rstart)
            r1 = rstart ;
         if (r1 > rstop)
            r1 = rstop ;
         r1 = b - r1 ;
         dc.MoveTo ( c0 , r0 ) ;
   		dc.LineTo ( c0 , r1 ) ;
         if (wide && (c0 > cstart)) {
            dc.MoveTo ( c0-1 , r0 ) ;
            dc.LineTo ( c0-1 , r1 ) ;
            }
         if (wide && (c0 < cstop)) {
            dc.MoveTo ( c0+1 , r0 ) ;
            dc.LineTo ( c0+1 , r1 ) ;
            }
         if ((wide > 9)  &&  (c0 > cstart+1)) {
            dc.MoveTo ( c0-2 , r0 ) ;
            dc.LineTo ( c0-2 , r1 ) ;
            }
         if ((wide > 9)  &&  (c0 < cstop-1)) {
            dc.MoveTo ( c0+2 , r0 ) ;
            dc.LineTo ( c0+2 , r1 ) ;
            }
   		}
      dc.RestorePen () ;
      }

	delete pen ;
	delete second_pen ;
	delete lines ;
}
