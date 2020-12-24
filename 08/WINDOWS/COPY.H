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

class CopyDialog: public TDialog {

public:
   CopyDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~CopyDialog () ;

   int parm_points, parm_start ;
   char source_name[256], dest_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   TButton *length ;
   TRadioButton *start, *end ;
   TEdit *points ;
   TRangeValidator *points_valid ;
   TListBox *source ;
   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvLength () ;
   void EvStart () ;
   void EvEnd () ;

	DECLARE_RESPONSE_TABLE ( CopyDialog ) ;
} ;

extern void do_copy ( TWindow *parent ) ;

