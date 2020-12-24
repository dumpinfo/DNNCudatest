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

class FilterDialog: public TDialog {

public:
   FilterDialog ( TWindow *parent , int p_detrend , int p_nsignals ,
                  Signal **p_signals ) ;
	~FilterDialog () ;

   int filter_type, use_detrend, parm_points, parm_order ;
   double parm_freq, parm_width ;
   char source_name[256], dest_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   TRadioButton *lowpass, *highpass, *bandpass, *savgol ;
   TRadioButton *ma, *qmf, *morlet ;
   TListBox *source ;
   TComboBox *dest ;
   TStatic *freq_label, *width_label, *points_label, *order_label ;
   TStatic *dest_label ;
   TEdit *freq, *width, *points, *order ;
   TCheckBox *detrend ;
   TFilterValidator *freq_valid, *width_valid ;
   TRangeValidator *points_valid, *order_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvLowpass () ;
   void EvHighpass () ;
   void EvBandpass () ;
   void EvSavgol () ;
   void EvMA () ;
   void EvQMF () ;
   void EvMorlet () ;
   void EvDetrend () ;

	DECLARE_RESPONSE_TABLE ( FilterDialog ) ;
} ;

extern void do_filter ( TWindow *parent ) ;

