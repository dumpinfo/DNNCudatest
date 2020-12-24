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

#if !defined(__prdctapp_h)              // Sentry, use file only if it's not already included.
#define __prdctapp_h

class npredictApp : public TApplication {
private:

public:
    npredictApp ();
    virtual ~npredictApp ();

public:
    virtual void InitMainWindow();

protected:
	 void CmReadSignal ();
	 void CmSaveSignal ();
	 void CmReadNetwork ();
	 void CmSaveNetwork ();
	 void CmReadARMA ();
	 void CmSaveARMA ();
	 void CmReadOrthog ();
	 void CmSaveOrthog ();
	 void CmCommandFile ();
	 void CmClear ();
	 void CmAuditLog () ;
	 void CmDisplaySignal ();
	 void CmGenerate () ;
	 void CmCopy () ;
	 void CmModify () ;
	 void CmFilter () ;
	 void CmLagCorr () ;
	 void CmSpectrum () ;
	 void CmArith () ;
	 void CmInputOutput ();
	 void CmCumulate () ;
	 void CmConfcomp () ;
	 void CmDefineOrthog () ;
	 void CmApplyOrthog () ;
	 void CmClass ();
	 void CmPrior ();
    void CmNetworkModel ();
	 void CmNetworkTrain ();
	 void CmNetworkTest ();
	 void CmNetworkPredict ();
    void CmARMAtrain ();
    void CmARMAshock ();
    void CmARMApredict ();
    void CmHelpAbout ();
    void CmHelpSize ();
DECLARE_RESPONSE_TABLE(npredictApp);
};
#endif                                      // __prdctapp_h sentry.
