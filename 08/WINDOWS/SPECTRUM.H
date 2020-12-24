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

class SpectrumDialog: public TDialog {

public:
   SpectrumDialog ( TWindow *parent , int p_window , int p_nsignals ,
                    Signal **p_signals ) ;
	~SpectrumDialog () ;

   int parm_type, parm_window, parm_points, parm_order ;
   char source_name[256], dest_name[256] ;

protected:

   int n_signals ;
   Signal **signals ;

   TRadioButton *dft, *maxent ;
   TRadioButton *nowindow, *welch ;
   TListBox *source ;
   TComboBox *dest ;
   TStatic *points_label, *order_label ;
   TStatic *dest_label ;
   TEdit *points, *order ;
   TRangeValidator *points_valid, *order_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvDft () ;
   void EvMaxent () ;
   void EvNoWindow () ;
   void EvWelch () ;

	DECLARE_RESPONSE_TABLE ( SpectrumDialog ) ;
} ;

extern void do_spectrum ( TWindow *parent ) ;

