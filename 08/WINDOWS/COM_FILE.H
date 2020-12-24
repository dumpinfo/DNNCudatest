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

#include "textlist.h"

class ComfileWindow : public TFrameWindow {
public:
   ComfileWindow ( TDecoratedMDIFrame *fr , char *title ) ;
   ~ComfileWindow () ;
	void AppendLine ( char *str ) ;
   int user_pressed_escape () ;
protected:
   void SetupWindow () ;
	void HandleAbortButton () ;
   void Paint ( TDC& dc , BOOL erase , TRect& badrec ) ;
private:
   TextList *lb ;              // AppendLine uses this
   int escape_key_pressed ;    // PeekMessage loop looks for ESCape key
	DECLARE_RESPONSE_TABLE ( ComfileWindow ) ;
} ;

extern void read_command_file ( TDecoratedMDIFrame *frame ) ;
