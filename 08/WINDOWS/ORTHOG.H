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

class OrthogDefineDialog: public TDialog {

public:
   OrthogDefineDialog ( TWindow *parent , int p_northogs , Orthog **p_orthogs );
	~OrthogDefineDialog () ;

   int parm_type, parm_factors, parm_std ;
   double parm_limit ;
   char orthog_name[256] ;


protected:

   int n_orthogs ;
   Orthog **orthogs ;

   TRadioButton *princo, *centroid, *discrim ;
   TCheckBox *std ;
   TEdit *factors, *limit ;
   TRangeValidator *factors_valid ;
   TFilterValidator *limit_valid ;
   TComboBox *orthog ;

   void SetupWindow () ;
   void CmOk () ;
   void EvPrinco () ;
   void EvCentroid () ;
   void EvDiscrim () ;
   void EvStd () ;

	DECLARE_RESPONSE_TABLE ( OrthogDefineDialog ) ;
} ;

extern void define_orthog ( TWindow *parent ) ;

/*
--------------------------------------------------------------------------------

   Apply

--------------------------------------------------------------------------------
*/

class OrthogApplyDialog: public TDialog {

public:
   OrthogApplyDialog ( TWindow *parent , int p_northogs , Orthog **p_orthogs  ,
                       int p_nsignals , Signal **p_signals);
	~OrthogApplyDialog () ;

   char orthog_name[256], dest_name[256] ;


protected:

   int n_signals ;
   Signal **signals ;

   int n_orthogs ;
   Orthog **orthogs ;

   TListBox *orthog ;
   TComboBox *dest ;

   void SetupWindow () ;
   void CmOk () ;

	DECLARE_RESPONSE_TABLE ( OrthogApplyDialog ) ;
} ;

extern void apply_orthog ( TWindow *parent ) ;
