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

#if !defined(__prdcmdic_h)              // Sentry, use file only if it's not already included.
#define __prdcmdic_h

class npredictMDIClient : public TMDIClient {
public:
    npredictMDIClient ();
    virtual ~npredictMDIClient ();

    int                     ChildCount;                 // Number of child window created. 

protected:
    virtual void SetupWindow ();

} ;
#endif                                      // __prdcmdic_h sentry.
