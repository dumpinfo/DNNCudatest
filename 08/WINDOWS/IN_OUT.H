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

#define MAX_WIN_INSOUTS 10

class InOutDialog: public TDialog {

public:
   InOutDialog ( TWindow *parent , int p_nsignals , Signal **p_signals ,
                 int p_ninsouts , InputOutput **p_insouts ) ;
	~InOutDialog () ;

   int nin, nout ;
   int in_mins[MAX_WIN_INSOUTS], in_maxs[MAX_WIN_INSOUTS] ;
   int out_mins[MAX_WIN_INSOUTS], out_maxs[MAX_WIN_INSOUTS] ;
   char in_names[MAX_WIN_INSOUTS][256], out_names[MAX_WIN_INSOUTS][256] ;

protected:

   int n_signals ;
   Signal **signals ;
   int n_insouts ;
   InputOutput **insouts ;

   TComboBox *in_name[MAX_WIN_INSOUTS], *out_name[MAX_WIN_INSOUTS] ;
   TEdit *in_min[MAX_WIN_INSOUTS], *in_max[MAX_WIN_INSOUTS] ;
   TEdit *out_min[MAX_WIN_INSOUTS], *out_max[MAX_WIN_INSOUTS] ;

   void SetupWindow () ;
   void CmOk () ;

	DECLARE_RESPONSE_TABLE ( InOutDialog ) ;
} ;

extern void ins_outs ( TWindow *parent ) ;

