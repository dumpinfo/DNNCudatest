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

class GenerateDialog: public TDialog {

public:
   GenerateDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ) ;
	~GenerateDialog () ;

   int parm_type, parm_points ;
   double parm_ar1, parm_ar2, parm_ar10, parm_ma1, parm_ma2, parm_ma10 ;
   double parm_mean, parm_std, parm_low, parm_high ;
   double parm_height, parm_rperiod, parm_amp, parm_speriod, parm_phase ;
   char dest_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   TRadioButton *Arma, *Normal, *Uniform, *Ramp, *Sine ;
   TStatic *ar1_label, *ar2_label, *ar10_label ;
   TStatic *ma1_label, *ma2_label, *ma10_label ;
   TEdit *ar1, *ar2, *ar10, *ma1, *ma2, *ma10 ;
   TFilterValidator *ar1_valid, *ar2_valid, *ar10_valid ;
   TFilterValidator *ma1_valid, *ma2_valid, *ma10_valid ;
   TStatic *mean_label, *std_label, *low_label, *high_label ;
   TStatic *height_label, *rperiod_label, *amp_label, *speriod_label, *phase_label ;
   TEdit *mean, *std, *low, *high, *height, *rperiod, *amp, *speriod, *phase ;
   TFilterValidator *mean_valid, *std_valid, *low_valid, *high_valid ;
   TFilterValidator *height_valid, *rperiod_valid, *amp_valid, *speriod_valid, *phase_valid ;
   TEdit *points ;
   TRangeValidator *points_valid ;
   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;
   void EvArma () ;
   void EvNormal () ;
   void EvUniform () ;
   void EvRamp () ;
   void EvSine () ;

	DECLARE_RESPONSE_TABLE ( GenerateDialog ) ;
} ;

extern void generate ( TWindow *parent ) ;

