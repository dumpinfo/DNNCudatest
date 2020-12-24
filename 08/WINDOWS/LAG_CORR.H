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

class LagCorrDialog: public TDialog {

public:
   LagCorrDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~LagCorrDialog () ;

   int corr_type, parm_maxlag ;
   char source_name[256], source2_name[256], dest_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   TRadioButton *autoc, *cross, *part_autoc, *part_cross ;
   TListBox *source, *source2 ;
   TComboBox *dest ;
   TStatic *source2_label ;
   TEdit *maxlag ;
   TRangeValidator *maxlag_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvAuto () ;
   void EvPartAuto () ;
   void EvCross () ;
   void EvPartCross () ;

	DECLARE_RESPONSE_TABLE ( LagCorrDialog ) ;
} ;

extern void lag_corr ( TWindow *parent ) ;

