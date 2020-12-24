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

class ModifyDialog: public TDialog {

public:
   ModifyDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~ModifyDialog () ;

   int parm_type, parm_median, parm_undo, parm_diff, parm_seasonal ;
   int parm_integrate ;
   double parm_offset, parm_scale ;
   char source_name[256], per_name[256] ;

protected:

   int n_signals ;
   Signal **signals ;

   TCheckBox *median, *undo ;
   TRadioButton *center, *detrend, *offset, *scale, *stdize ;
   TRadioButton *diff, *seasonal, *log, *exp, *integrate ;
   TEdit *offset_parm, *scale_parm, *diff_parm, *seasonal_parm, *integrate_parm;
   TStatic *source_label, *per_label ;
   TListBox *source, *per ;

   TFilterValidator *offset_valid, *scale_valid ;
   TRangeValidator *diff_valid, *seasonal_valid, *integrate_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvUndo () ;
   void EvMedian () ;
   void EvCenter () ;
   void EvDetrend () ;
   void EvOffset () ;
   void EvScale () ;
   void EvStdize () ;
   void EvDiff () ;
   void EvSeasonal () ;
   void EvLog () ;
   void EvExp () ;
   void EvIntegrate () ;

	DECLARE_RESPONSE_TABLE ( ModifyDialog ) ;
} ;

extern void do_modify ( TWindow *parent ) ;

