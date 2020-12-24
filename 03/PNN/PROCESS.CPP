// Copyright (c) 1994 John Wiley & Sons, Inc.  All rights reserved.
// Reproduction or translation of this work beyond that permitted in
// section 117 of the 1976 United States Copyright Act without the
// express written permission of the copyright owner is unlawful.
// Requests for further information should be addressed to the
// Permission Department, John Wiley & Sons, Inc.  The purchaser may
// make backup copies for his/her own use only and not for distribution
// or resale.  The publisher assumes no responsibility for errors,
// omissions, or damages, caused by the use of these programs or from
// the use of the information contained herein.

/******************************************************************************/
/*                                                                            */
/*  PROCESS - Process control commands                                        */
/*                                                                            */
/*  This returns 1 if user wants to quit, else 0.                             */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"       // System and limitation constants, typedefs, structs
#include "classes.h"     // Includes all class headers
#include "funcdefs.h"    // Function prototypes

#define PNN_CBLOCK_SIZE 2048

/*
   Forward declarations for static routines
*/

static int ok_to_clear_tset ( TrainingSet **tset , PNNet **network ) ;
static int ok_to_clear_network ( PNNet **network ) ;

int process ( int id , char *rest , NetParams *net_params ,
              LearnParams *learn_params , MiscParams *misc_params ,
              ControlData *cbuf , PNNet **network , TrainingSet **tset )

{
   int m, n, istart, avail ;
   double p ;
   char *cdata ;               // Control file read into this buffer
   char msg[256] ;
   FILE *fp ;

/*
   Act on the command
*/

   if (! id)      // NULL input line or other simple problem
      return 0 ;

   if (id == ID_PNNCOM_BYE)
      return 1 ;

   if (id == ID_PNNCOM_COMMANDFILE) {
      fp = fopen ( rest , "rt" ) ;
      if (fp == NULL) {
         printf ( "\nERROR... Cannot open %s.", rest ) ;
         return 0 ;
         }

      MEMTEXT ( "PROCESS: Initial cdata" ) ;
      cdata = (char *) MALLOC ( PNN_CBLOCK_SIZE ) ;
      if (cdata == NULL) {
         memory_message ( "to read COMMAND FILE" ) ;
         return 0 ;
         }

      istart = 0 ;
      avail = PNN_CBLOCK_SIZE ;
      for (;;) {
         if (avail < PNN_CBLOCK_SIZE / 4) { // Need more?
            MEMTEXT ( "PROCESS: Expand cdata" ) ;
            cdata = (char *) REALLOC ( cdata , istart + PNN_CBLOCK_SIZE ) ;
            if (cdata == NULL) {
               memory_message ( "to read COMMAND FILE" ) ;
               return 0 ;
               }
            avail = PNN_CBLOCK_SIZE ;
            }
         if (fgets ( cdata+istart , avail , fp ) == NULL ) { //EOF?
            fclose ( fp ) ; // If so, close it
            break ;
            }

         n = strlen ( cdata+istart ) ;   // fgets returns the newline at end
         cdata[istart+n-1] = 0 ;         // so we must remove it

         if (n > 1) {                    // It may be that it was a null line
            istart += n ;                // If good, keep it
            avail -= n ;                 // This much less available in cdata
            }
         }

      if (! istart) {
         error_message ( "No data in COMMAND FILE" ) ;
         MEMTEXT ( "PROCESS: Error free cdata" ) ;
         FREE ( cdata ) ;
         return 0 ;
         }

      MEMTEXT ( "PROCESS: Final cdata" ) ;
      cdata = (char *) REALLOC ( cdata , istart + cbuf->n ) ;
      if (cbuf->n) {  // If there is still unprocessed data in the buffer
         memcpy ( cdata+istart , cbuf->buf , cbuf->n ) ; // Append it
         MEMTEXT ( "PROCESS: free old buf" ) ;
         FREE ( cbuf->buf ) ;
         }

      cbuf->buf = cdata ;
      cbuf->n += istart ;
      return 0 ;
      }

   if (id == ID_PNNCOM_NETMOD) {
      if (! strcmp ( rest , "BASIC" ))
         n = NETMOD_BASIC ;
      else if (! strcmp ( rest , "SEPVAR" ))
         n = NETMOD_SEPVAR ;
      else if (! strcmp ( rest , "SEPCLASS" ))
         n = NETMOD_SEPCLASS ;
      else if (! strcmp ( rest , "GCNN" ))
         n = NETMOD_GCNN ;
      else if (! strcmp ( rest , "GCNN_EW" ))
         n = NETMOD_GCNN_EW ;
      else {
         sprintf ( msg , "Illegal NETWORK MODEL = %s", rest ) ;
         error_message ( msg ) ;
         return 0 ;
         }
      if (net_params->net_model == n)
         return 0 ;
      if (ok_to_clear_network( network ))
         net_params->net_model = n ;
      else
         warning_message ( "Command aborted" ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_KERNEL) {
      if (! strcmp ( rest , "GAUSS" ))
         n = KERNEL_GAUSS ;
      else if (! strcmp ( rest , "RECIPROCAL" ))
         n = KERNEL_RECIP ;
      else {
         sprintf ( msg , "Illegal KERNEL = %s", rest ) ;
         error_message ( msg ) ;
         return 0 ;
         }
      if (net_params->net_kernel == n)
         return 0 ;
      if (ok_to_clear_network( network ))
         net_params->net_kernel = n ;
      else 
         warning_message ( "Command aborted" ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_OUTMOD) {
      if (! strcmp ( rest , "CLASSIFICATION" ))
         n = OUTMOD_CLASSIFICATION ;
      else if (! strcmp ( rest , "AUTOASSOCIATION" ))
         n = OUTMOD_AUTOASSOCIATION ;
      else if (! strcmp ( rest , "MAPPING" ))
         n = OUTMOD_MAPPING ;
      else {
         sprintf ( msg , "Illegal MODE = %s", rest ) ;
         error_message ( msg ) ;
         return 0 ;
         }
      if (net_params->out_model == n)
         return 0 ;
      if ((ok_to_clear_tset( tset , network )) &&
          (ok_to_clear_network( network )))
         net_params->out_model = n ;
      else
         warning_message ( "Command aborted" ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_NIN) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal INPUTS = %s", rest ) ;
         error_message ( msg ) ;
         return 0 ;
         }
      if (net_params->n_inputs == n)
         return 0 ;
      if ((ok_to_clear_tset( tset , network )) &&
          (ok_to_clear_network(network)))
         net_params->n_inputs = n ;
      else
         warning_message ( "Command aborted" ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_NOUT) {
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal OUTPUTS = %s", rest ) ;
         error_message ( msg ) ;
         return 0 ;
         }
      if (net_params->n_outputs == n)
         return 0 ;
      if ((ok_to_clear_tset( tset , network )) &&
          (ok_to_clear_network(network)))
         net_params->n_outputs = n ;
      else
         warning_message ( "Command aborted" ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_TRAIN) {
      if ((net_params->out_model == OUTMOD_AUTOASSOCIATION)  &&
          (net_params->n_outputs != net_params->n_inputs)) {
         error_message (
                  "AUTOASSOCIATIVE mode but OUTPUTS not equal to INPUTS" ) ;
         return 0 ;
         }
      if (net_params->out_model <= 0)
         error_message ( "CUMULATE TRAINING SET used before MODE set." ) ;
      else if (net_params->n_inputs <= 0)
         error_message ( "CUMULATE TRAINING SET used before INPUTS set." ) ;
      else if (net_params->n_outputs <= 0)
         error_message ( "CUMULATE TRAINING SET used before OUTPUTS set." ) ;
      else {
         if (*tset == NULL) {
            MEMTEXT ( "PNN: new tset" ) ;
            *tset = new TrainingSet ( net_params->out_model ,
                                      net_params->n_inputs ,
                                      net_params->n_outputs ) ;
            if (*tset == NULL) {
               memory_message ( "to create training set" ) ;
               return 0 ;
               }
            }
         if (((*network) != NULL)  &&  (*network)->privatize_tset ()) {
            if (!get_yn("Low memory.  Trained network must be cleared.  OK")) {
               warning_message ( "Command aborted" ) ;
               return 0 ;
               }
            MEMTEXT ( "PNN: delete network to modify shared tset" ) ;
            delete *network ;
            *network = NULL ;
            }
         (*tset)->train ( rest , misc_params->classif_output ,
                          misc_params->classif_prior ) ;
         }
      return 0 ;
      }

   if (id == ID_PNNCOM_QUITERR) {
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( msg , "Illegal ALLOWABLE ERROR = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->quit_err = p ;
      return 0 ;
      }

   if (id == ID_PNNCOM_QUITTOL) {
      p = atof ( rest ) ;
      if (p <= 0.0) {
         sprintf ( msg , "Illegal TOLERANCE = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->quit_tol = p ;
      return 0 ;
      }

   if (id == ID_PNNCOM_SIGLO) {
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( msg , "Illegal SIGMA LOW = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->siglo = p ;
      return 0 ;
      }

   if (id == ID_PNNCOM_SIGHI) {
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( msg , "Illegal SIGMA HIGH = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->sighi = p ;
      return 0 ;
      }

   if (id == ID_PNNCOM_SIGTRIES) {
      n = atoi ( rest ) ;
      if (n < 3) {
         sprintf ( msg , "Illegal SIGMA TRIES = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         learn_params->nsigs = n ;
      return 0 ;
      }

   if (id == ID_PNNCOM_LEARN) {
      if ((*tset == NULL)  ||  ((*tset)->ntrain == 0)) {
         error_message ( "Cannot COMPUTE WEIGHTS; No training set exists." ) ;
         return 0 ;
         }
      if (net_params->out_model == OUTMOD_CLASSIFICATION) {
         for (m=0 ; m<net_params->n_outputs ; m++) {
            if ((*tset)->nper[m] == 0) {
               error_message("Cannot COMPUTE WEIGHTS; A class has 0 cases.") ;
               return 0 ;
               }
            }
         }
      if (((net_params->net_model == NETMOD_BASIC)  ||
           (net_params->net_model == NETMOD_SEPVAR) ||
           (net_params->net_model == NETMOD_SEPCLASS))  &&
           (learn_params->siglo > learn_params->sighi)) {
         error_message("Cannot COMPUTE WEIGHTS; SIGMA LOW exceeds SIGMA HIGH.");
         return 0 ;
         }
      if ((net_params->net_model == NETMOD_SEPVAR)  ||
          (net_params->net_model == NETMOD_SEPCLASS)) {
         if (net_params->net_kernel != KERNEL_GAUSS) {
            error_message( "Cannot COMPUTE WEIGHTS; SEP... for GAUSS only." ) ;
            return 0 ;
            }
         }
      if (net_params->net_model == NETMOD_GCNN) {
         if (net_params->out_model != OUTMOD_CLASSIFICATION) {
            error_message(
               "Cannot COMPUTE WEIGHTS; GCNN for CLASSIFICATION only." ) ;
            return 0 ;
            }
         if (net_params->maxmom > 8) {
            error_message( "Cannot COMPUTE WEIGHTS; GCNN max moments > 8." ) ;
            return 0 ;
            }
         }
      if (net_params->net_model == NETMOD_GCNN_EW) {
         if (net_params->out_model != OUTMOD_CLASSIFICATION) {
            error_message(
               "Cannot COMPUTE WEIGHTS; GCNN_EW for CLASSIFICATION only." ) ;
            return 0 ;
            }
         if (net_params->maxmom > 5) {
            error_message( "Cannot COMPUTE WEIGHTS; GCNN_EW max moments > 5." );
            return 0 ;
            }
         }
      if (*network == NULL) {
         if (((net_params->net_model == NETMOD_BASIC)  ||
              (net_params->net_model == NETMOD_SEPVAR) ||
              (net_params->net_model == NETMOD_SEPCLASS))  &&
              (net_params->net_kernel <= 0)) {
            error_message ( "Cannot COMPUTE WEIGHTS; KERNEL not set." ) ;
            return 0 ;
            }
         MEMTEXT ( "PNN: new PNNet" ) ;
         if (net_params->net_model == NETMOD_BASIC)
            *network = new PNNbasic ( net_params->net_kernel ,
                                      net_params->out_model ,
                                      net_params->n_inputs ,
                                      net_params->n_outputs ) ;
         else if (net_params->net_model == NETMOD_SEPVAR)
            *network = new PNNsepvar ( net_params->net_kernel ,
                                       net_params->out_model ,
                                       net_params->n_inputs ,
                                       net_params->n_outputs ) ;
         else if (net_params->net_model == NETMOD_SEPCLASS)
            *network = new PNNsepclass ( net_params->net_kernel ,
                                         net_params->out_model ,
                                         net_params->n_inputs ,
                                         net_params->n_outputs ) ;
         else if (net_params->net_model == NETMOD_GCNN)
            *network = new GCNN ( net_params->maxmom , 0 ,
                                  net_params->out_model ,
                                  net_params->n_inputs ,
                                  net_params->n_outputs ) ;
         else if (net_params->net_model == NETMOD_GCNN_EW)
            *network = new GCNN ( net_params->maxmom , 1 ,
                                  net_params->out_model ,
                                  net_params->n_inputs ,
                                  net_params->n_outputs ) ;
         else {
            error_message ( "Cannot COMPUTE WEIGHTS; NETWORK MODEL not set." ) ;
            return 0 ;
            }
         }
      if ((*network == NULL)  ||  (! (*network)->ok)) {  // Malloc failure?
         memory_message ( "to create network." ) ;
         if (*network != NULL) {
            delete *network ;
            *network = NULL ;
            }
         return 0 ;
         }
      (*network)->learn ( *tset , learn_params ) ;
      if (! (*network)->errtype) {  // Indicates massive failure
         MEMTEXT ( "PNN: learn failure delete network" ) ;
         delete *network ;
         *network = NULL ;
         }
      else {
         sprintf ( msg , "Final error = %.6lf", (*network)->neterr ) ;
         normal_message ( msg ) ;
         }
      return 0 ;
      }

   if (id == ID_PNNCOM_SAVENET) {
      if (*network == NULL)
         error_message ( "There is no trained network to save." ) ;
      else
         wt_save ( *network , 0 , rest ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_RESTNET) {
      if (*network != NULL) {
         MEMTEXT ( "PNN: delete network for restore" ) ;
         delete *network ;
         *network = NULL ;
         }
      *network = wt_restore ( rest ) ;
      if (*network == NULL)
         return 0 ;
      if (*tset != NULL) {
         if ((net_params->n_inputs != (*network)->nin)
          || (net_params->n_outputs != (*network)->nout)
          || (net_params->out_model != (*network)->outmod)) {
            error_message ( "Net conflicts with existing training set." ) ;
            MEMTEXT ( "PNN: delete network" ) ;
            delete *network ;
            *network = NULL ;
            return 0 ;
            }
         }
      net_params->net_model = (*network)->model ;
      net_params->net_kernel = (*network)->kernel ;
      net_params->maxmom = (*network)->maxmom ;
      net_params->out_model = (*network)->outmod ;
      net_params->n_inputs = (*network)->nin ;
      net_params->n_outputs = (*network)->nout ;
      return 0 ;
      }

   if (id == ID_PNNCOM_CLEARTRAIN) {
      if (*tset != NULL) {
         if (((*network) != NULL)  &&  (*network)->privatize_tset ()) {
            if (!get_yn("Low memory. Trained network must also be cleared. OK"))
               return 0 ;
            MEMTEXT ( "PNN: delete network along with shared tset" ) ;
            delete *network ;
            *network = NULL ;
            }
         MEMTEXT ( "PNN: delete tset" ) ;
         delete *tset ;
         *tset = NULL ;
         }
      return 0 ;
      }

   if (id == ID_PNNCOM_CLEARNET) {
      if (*network != NULL) {
         MEMTEXT ( "PNN: delete network" ) ;
         delete *network ;
         *network = NULL ;
         }
      return 0 ;
      }

   if (id == ID_PNNCOM_CLASSOUT) {
      if (net_params->n_outputs < 0) {
         error_message ( "CLASS used before OUTPUTS set." ) ;
         return 0 ;
         }
      if (net_params->out_model != OUTMOD_CLASSIFICATION) {
         error_message
               ( "CLASS only valid when MODE=CLASSIFICATION" ) ;
         return 0 ;
         }
      m = sscanf ( rest , "%d" , &n ) ;
      if ((m <= 0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal CLASS = %s", rest ) ;
         error_message ( msg ) ;
         }
      else if (n > net_params->n_outputs) {
         sprintf ( msg , "CLASS (%d) exceeds OUTPUTS (%d)",
                   n, net_params->n_outputs ) ;
         error_message ( msg ) ;
         }
      else
         misc_params->classif_output = n ;
      return 0 ;
      }

   if (id == ID_PNNCOM_PRIOR) {
      if (net_params->out_model != OUTMOD_CLASSIFICATION) {
         error_message
               ( "PRIOR only valid when MODE=CLASSIFICATION" ) ;
         return 0 ;
         }
      if (! strcmp ( rest , "N" )) {
         misc_params->classif_prior = -1.0 ;      // Flag to use class count
         return 0 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( msg , "Illegal PRIOR = %s", rest ) ;
         error_message ( msg ) ;
         }
      else
         misc_params->classif_prior = p ;
      return 0 ;
      }

   if (id == ID_PNNCOM_OUTFILE) {
      strcpy ( misc_params->out_file , rest ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_EXECUTE) {
      if (*network == NULL)
         error_message ( "There is no trained network" ) ;
      else
         (*network)->execute_from_file ( rest , misc_params->out_file ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_TEST) {
      if (*network == NULL)
         error_message ( "There is no trained network" ) ;
      else if (net_params->out_model == OUTMOD_CLASSIFICATION)
         error_message ( "TEST not valid in CLASSIFICATION mode" ) ;
      else
         (*network)->test_from_file ( rest ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_CLASSIFY) {
      if (*network == NULL)
         error_message ( "There is no trained network" ) ;
      else if (net_params->out_model != OUTMOD_CLASSIFICATION)
         error_message ( "CLASSIFY valid only in CLASSIFICATION mode" ) ;
      else
         (*network)->classify_from_file ( rest , misc_params->threshold ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_RESET) {
      if (*network == NULL)
         error_message ( "There is no trained network" ) ;
      else if (net_params->out_model != OUTMOD_CLASSIFICATION)
         error_message ( "CONFUSION valid only in CLASSIFICATION mode" ) ;
      else
         (*network)->reset_confusion () ;
      return 0 ;
      }

   if (id == ID_PNNCOM_THRESH) {
      p = atof ( rest ) ;
      if ((p < 0.0)  ||  (p > 100.0)) {
         sprintf( msg, "Illegal ACTIVATION THRESHOLD FOR CONFUSION = %s", rest);
         error_message ( msg ) ;
         }
      else
         misc_params->threshold = p / 100.0 ;
      return 0 ;
      }

   if (id == ID_PNNCOM_SHOWCON) {
      if (*network == NULL)
         error_message ( "There is no trained network" ) ;
      else if (net_params->out_model != OUTMOD_CLASSIFICATION)
         error_message ( "CONFUSION valid only in CLASSIFICATION mode" ) ;
      else
         (*network)->show_confusion () ;
      return 0 ;
      }

   if (id == ID_PNNCOM_SAVECON) {
      if (*network == NULL)
         error_message ( "There is no trained network" ) ;
      else if (net_params->out_model != OUTMOD_CLASSIFICATION)
         error_message ( "CONFUSION valid only in CLASSIFICATION mode" ) ;
      else
         (*network)->save_confusion ( rest ) ;
      return 0 ;
      }

   if (id == ID_PNNCOM_MAXMOM) {
      n = atoi ( rest ) ;
      if (n < 3) {
         sprintf( msg, "Illegal MAX MOMENT = %s", rest);
         error_message ( msg ) ;
         }
      else if (net_params->maxmom == n)
         return 0 ;
      else if (ok_to_clear_network( network ))
         net_params->maxmom = n ;
      else
         warning_message ( "Command aborted" ) ;
      return 0 ;
      }

   return 0 ;
}

/*
--------------------------------------------------------------------------------

   Static routines

--------------------------------------------------------------------------------
*/

static int ok_to_clear_tset( TrainingSet **tset , PNNet **network )
{
   if (*tset == NULL)
      return 1 ;

   if (get_yn ( "Existing training set must be cleared.  OK" )) {
      if (((*network) != NULL)  &&  (*network)->privatize_tset ()) {
         if (! get_yn("Low memory.  Trained network must also be cleared. OK"))
            return 0 ;
         MEMTEXT ( "PNN: delete network along with shared tset" ) ;
         delete *network ;
         *network = NULL ;
         }
      MEMTEXT ( "PNN: delete tset" ) ;
      delete *tset ;
      *tset = NULL ;
      return 1 ;
      }
   else
      return 0 ;
}

static int ok_to_clear_network( PNNet **network )
{
   if (*network == NULL)
      return 1 ;

   if (get_yn ( "Existing trained network must be cleared.  OK" )) {
      MEMTEXT ( "PNN: delete network" ) ;
      delete *network ;
      *network = NULL ;
      return 1 ;
      }
   else
      return 0 ;
}
