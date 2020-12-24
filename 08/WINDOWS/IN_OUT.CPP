/******************************************************************************/
/*                                                                            */
/*  IN_OUT - Build input and output lists                                     */
/*                                                                            */
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
/*                                                                            */
/******************************************************************************/

#include <owl\owlpch.h>
#include <\owl\listbox.h>
#include <\owl\button.h>
#include <\owl\static.h>
#include <\owl\validate.h>
#include <\owl\groupbox.h>
#include <owl\inputdia.h>
#include <owl\dialog.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#pragma hdrstop

#include "..\common\const.h"  // System, limitation constants, typedefs, structs
#include "..\common\classes.h"  // Includes all class headers
#include "..\common\funcdefs.h" // Function prototypes

#include "in_out.h"
#include "prdctapp.rh"

DEFINE_RESPONSE_TABLE1 ( InOutDialog , TDialog )
  EV_COMMAND ( IDOK , CmOk ) ,
END_RESPONSE_TABLE ;

InOutDialog::InOutDialog ( TWindow *parent ,
                           int p_nsignals , Signal **p_signals ,
                           int p_ninsouts , InputOutput **p_insouts )
	: TDialog ( parent , IDD_INSOUTS ), TWindow ( parent )
{
   int i ;
   TRangeValidator *valid ;

   n_signals = p_nsignals ;
   signals = p_signals ;
   n_insouts = p_ninsouts ;
   insouts = p_insouts ;

   for (i=0 ; i<MAX_WIN_INSOUTS ; i++) {
      in_name[i] = new TComboBox ( this , IDC_INSOUTS_INNAME+i ) ;
      in_min[i] = new TEdit ( this , IDC_INSOUTS_INMIN+i ) ;
      valid = new TRangeValidator ( 0 , 999999 ) ;
      in_min[i]->SetValidator ( valid ) ;
      in_max[i] = new TEdit ( this , IDC_INSOUTS_INMAX+i ) ;
      valid = new TRangeValidator ( 0 , 999999 ) ;
      in_max[i]->SetValidator ( valid ) ;
      out_name[i] = new TComboBox ( this , IDC_INSOUTS_OUTNAME+i ) ;
      out_min[i] = new TEdit ( this , IDC_INSOUTS_OUTMIN+i ) ;
      valid = new TRangeValidator ( 0 , 999999 ) ;
      out_min[i]->SetValidator ( valid ) ;
      out_max[i] = new TEdit ( this , IDC_INSOUTS_OUTMAX+i ) ;
      valid = new TRangeValidator ( 0 , 999999 ) ;
      out_max[i]->SetValidator ( valid ) ;
      }
}

InOutDialog::~InOutDialog ()
{
}

void InOutDialog::SetupWindow ()
{
   int i, j ;
   char msg[84] ;
   Signal *sigptr ;
   TComboBox *in, *out ;
   InputOutput *ioptr ;

	TDialog::SetupWindow () ;

   for (i=0 ; i<MAX_WIN_INSOUTS ; i++) {
      in = in_name[i] ;
      out = out_name[i] ;
      for (j=0 ; j<n_signals ; j++) {
         sigptr = signals[j] ;
         in->AddString ( sigptr->name ) ;
         out->AddString ( sigptr->name ) ;
         }
      in->SetSelIndex ( -1 ) ;
      out->SetSelIndex ( -1 ) ;
      in_min[i]->SetText ( "0" ) ;
      in_max[i]->SetText ( "0" ) ;
      out_min[i]->SetText ( "0" ) ;
      out_max[i]->SetText ( "0" ) ;
      }

   nin = nout = 0 ;
   for (i=0 ; i<n_insouts ; i++) {
      ioptr = insouts[i] ;
      if (ioptr->is_input) {
         if (nin == MAX_WIN_INSOUTS)
            continue ;
         j = in_name[nin]->FindExactString ( signals[ioptr->which]->name , 0 ) ;
         in_name[nin]->SetSelIndex ( j ) ;
         sprintf ( msg , "%d" , ioptr->minlag ) ;
         in_min[nin]->SetText ( msg ) ;
         sprintf ( msg , "%d" , ioptr->maxlag ) ;
         in_max[nin]->SetText ( msg ) ;
         ++nin ;
         }
      else {
         if (nout == MAX_WIN_INSOUTS)
            continue ;
         j = out_name[nout]->FindExactString ( signals[ioptr->which]->name , 0 ) ;
         out_name[nout]->SetSelIndex ( j ) ;
         sprintf ( msg , "%d" , ioptr->minlag ) ;
         out_min[nout]->SetText ( msg ) ;
         sprintf ( msg , "%d" , ioptr->maxlag ) ;
         out_max[nout]->SetText ( msg ) ;
         ++nout ;
         }
      }

}

void InOutDialog::CmOk ()
{
   int i ;
   char msg[256] ;

   for (i=0 ; i<MAX_WIN_INSOUTS ; i++) {
      in_name[i]->GetText ( in_names[i] , 255 ) ;
      strupr ( in_names[i] ) ;
      out_name[i]->GetText ( out_names[i] , 255 ) ;
      strupr ( out_names[i] ) ;
      in_min[i]->GetText ( msg , 255 ) ;
      in_mins[i] = atoi ( msg ) ;
      in_max[i]->GetText ( msg , 255 ) ;
      in_maxs[i] = atoi ( msg ) ;
      out_min[i]->GetText ( msg , 255 ) ;
      out_mins[i] = atoi ( msg ) ;
      out_max[i]->GetText ( msg , 255 ) ;
      out_maxs[i] = atoi ( msg ) ;
      }

	TDialog::CmOk () ;
}

/*
--------------------------------------------------------------------------------

	This is a global routine called to do the busywork

--------------------------------------------------------------------------------
*/

void ins_outs ( TWindow *parent )
{
   int i, nsigs, nio ;
	char error[256], rest[256] ;
	InOutDialog *iod ;
   Signal **signals ;
   InputOutput **io ;

   nsigs = get_signals ( &signals ) ;
   nio = get_ins_outs ( &io ) ;
	iod = new InOutDialog ( parent , nsigs , signals , nio , io ) ;

	if (iod->Execute() == IDOK) {
      process ( ID_PRED_CLEAR_INPUT , NULL , NULL , error , NULL ) ;
      process ( ID_PRED_CLEAR_OUTPUT , NULL , NULL , error , NULL ) ;
      for (i=0 ; i<MAX_WIN_INSOUTS ; i++) {
         if (strlen ( iod->in_names[i] )) {
            sprintf ( rest , "%s %d %d", iod->in_names[i], iod->in_mins[i],
                                                           iod->in_maxs[i] ) ;
            if (process ( ID_PRED_INPUT , rest , NULL , error , NULL ) < 0){
               parent->MessageBox ( error , "ERROR" ) ;
               delete iod ;
               return ;
               }
            }
         }
      for (i=0 ; i<MAX_WIN_INSOUTS ; i++) {
         if (strlen ( iod->out_names[i] )) {
            sprintf ( rest , "%s %d %d", iod->out_names[i], iod->out_mins[i],
                                                            iod->out_maxs[i] ) ;
            if (process ( ID_PRED_OUTPUT , rest , NULL , error , NULL ) < 0) {
               parent->MessageBox ( error , "ERROR" ) ;
               delete iod ;
               return ;
               }
            }
         }
      }

	delete iod ;
}

