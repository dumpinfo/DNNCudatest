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

class ProgressWindow : public TFrameWindow {
public:
   ProgressWindow ( char *title ) ;
   ~ProgressWindow () ;
	void AppendLine ( char *str ) ;
protected:
   void SetupWindow () ;
	void HandleAbortButton () ;
   void Paint ( TDC& dc , BOOL erase , TRect& badrec ) ;
	DECLARE_RESPONSE_TABLE ( ProgressWindow ) ;
} ;

extern void init_progress ( TDecoratedMDIFrame *frame ) ;
extern void make_progress_window ( char *title ) ;
extern void destroy_progress_window () ;
extern void write_progress () ;
extern int user_pressed_escape () ;
