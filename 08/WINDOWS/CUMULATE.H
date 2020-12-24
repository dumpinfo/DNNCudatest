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

class CumulateDialog: public TDialog {

public:
   CumulateDialog ( TWindow *parent ) ;
	~CumulateDialog () ;

   int parm_type, parm_include, parm_exclude ;


protected:

   TRadioButton *training, *test ;
   TEdit *include, *exclude ;
   TRangeValidator *include_valid, *exclude_valid ;

   void SetupWindow () ;
   void CmOk () ;
   void EvTraining () ;
   void EvTest () ;

	DECLARE_RESPONSE_TABLE ( CumulateDialog ) ;
} ;

extern void cumulate ( TWindow *parent ) ;
