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

class ConfCompDialog: public TDialog {

public:
   ConfCompDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~ConfCompDialog () ;

   int parm_clear ;
   enum ConfCompType parm_type ;
   char parm_sig[256], parm_per[256] ;

protected:

   int n_signals ;
   Signal **signals ;

   TCheckBox *clear ;
   TRadioButton *center, *detrend, *offset, *scale, *std, *log, *diff,*seasonal;
   TStatic *per_label ;
   TListBox *sig ;
   TComboBox *per ;

   void SetupWindow () ;
   void CmOk () ;
   void EvClear () ;
   void EvCenter () ;
   void EvDetrend () ;
   void EvOffset () ;
   void EvScale () ;
   void EvStd () ;
   void EvLog () ;
   void EvDiff () ;
   void EvSeasonal () ;

	DECLARE_RESPONSE_TABLE ( ConfCompDialog ) ;
} ;

extern void set_conf_comp ( TWindow *parent ) ;

