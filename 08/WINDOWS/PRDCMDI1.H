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

#if !defined(__prdcmdi1_h)              // Sentry, use file only if it's not already included.
#define __prdcmdi1_h

#include "..\common\const.h"
#include "..\common\classes.h"
#include "prdctapp.rh"            // Definition of all resources.

class npredictMDIChild : public TMDIChild {
public:
	npredictMDIChild (TMDIClient *parent, const char far *title,
							TWindow *clientWnd = 0, BOOL shrinkToClient = FALSE,
							TModule* module = 0 ) ;
	virtual ~npredictMDIChild ();
	
	Signal *signal ;  // The Signal in this window
   int command_id ;  // Command id that created this window
	int xndigits, yndigits, xnfrac, ynfrac ; // Label information
	int xnticks, ynticks ; // Number of tick marks
   int istart, istop ;    // First and last (org 0) points to display
	double xmin, xmax, xdif, ymin, ymax, ydif, leftx, rightx ; // Graph limits
   double corrlim ;       // Confidence limits for CORRELATION plot
	COLORREF background_color ;

protected:
	 virtual void Paint ( TDC& dc , BOOL erase , TRect& rect ) ;
	 void EvSize ( UINT sizeType , TSize &size ) ;
	DECLARE_RESPONSE_TABLE ( npredictMDIChild ) ;
};

extern void purge_display ( npredictMDIChild *child ) ;

#endif                                      // __prdcmdi1_h sentry.
