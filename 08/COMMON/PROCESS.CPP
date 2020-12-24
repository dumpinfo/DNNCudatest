/******************************************************************************/
/*                                                                            */
/*  PROCESS - Process control commands                                        */
/*                                                                            */
/*  This returns 1 if user wants to quit, -1 if error, else 0.                */
/*  If there was an error, the message is returned in 'error'                 */
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

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>
#include <ctype.h>
#include <stdlib.h>
#include "const.h"     // System, limitation constants, typedefs, structs
#include "classes.h"   // Includes all class headers
#include "funcdefs.h"  // Function prototypes

#define PREDICT_CBLOCK_SIZE 2048

static int alphabetize () ;
static int check_arma_compatibility ( int nweights , int nout , int arma_fixed ,
                                      ARMA *arma , char *error ) ;
static int check_learn_params ( char *error ) ;
static int check_net_params ( char *error ) ;
static int check_test_set ( char *error ) ;
static int check_training_set ( char *error ) ;
static int check_set_net_compatibility ( TrainingSet *trnset , Network *net ,
                                         char *error ) ;
static int check_set_net_inputs ( TrainingSet *trnset , Network *net ,
                                  char *error ) ;
static int check_set_net_outputs ( TrainingSet *trnset , Network *net ,
                                  char *error ) ;
static void clear_all () ;

/*
   These are network parameters that may be set by the user via commands.
   Most are initialized to defaults which indicate that the user has not
   yet set them.  As they are set, their current values are placed here.
   When learning is performed for a network, their values are copied from here
   into the network object.  When a network is read, the object's values
   are copied from it to here.  Otherwise, these variables are not used;
   the values in the network object itself are used.  The only purpose of
   these variables is to keep track of current values as specified by the user.
*/

static struct NetParams net_params ;         // Network model parameters

/*
   In a similar vein, these are other parameters.  They are initialized in
   DEFAULTS.CPP, and they are reset by various user commands.
*/

static struct LearnParams learn_params ;     // General learning parameters
static struct AnnealParams anneal_params ;   // Simulated annealing parameters
static struct MiscParams misc_params ;       // Other parameters

/*
   These are the current default inputs/outputs for training set construction
*/

static int n_ins_outs = 0 ;
static InputOutput **ins_outs = NULL ;
static int n_classes = 0 ;
static char **classes = NULL ;

/*
   All signals are kept here.
*/

static int n_signals = 0 ;
static Signal **signals = NULL ;

/*
   This is the training and test set
*/

TrainingSet *trnset = NULL ;
TrainingSet *testset = NULL ;

/*
   All neural networks are kept here.
*/

static int n_networks = 0 ;
static Network **networks = NULL ;

/*
   All ARMA models are kept here.
*/

static int n_ARMAs = 0 ;
static ARMA **ARMAs = NULL ;
static int arma_fixed = 0 ;

/*
   All orthogonalizations are here
*/

static int n_orthogs = 0 ;
static Orthog **orthogs = NULL ;

/*
   All confidence compensation records
*/

static int n_conf_comps = 0 ;
static ConfComp *conf_comps = NULL ;


static char audit_log[256] = { "AUDIT.LOG" } ;

/*
--------------------------------------------------------------------------------

   This is the entry point.

--------------------------------------------------------------------------------
*/

int process ( int id , char *rest , ControlData *cbuf , char *error ,
              char *results )

{
   int i, j, m, n, ncases, istart, avail, ret, n_inputs, n_outputs ;
   int out_model, n_ma, *confusion, iout, degree, period, outsig, shocksig ;
   double p, t, confuse, *wptr, *dptr, sum, freq, width ;
   char *cdata ;               // Control file read into this buffer
   char **cptr ;
   char netname[84], armaname[84], orthname[84] ;
   char filename[256], msg[84], name[84] ;
   FILE *fp ;
   Signal *sigptr, *sigptr2 ;
   TrainingSet **tset ;
   Network *net, **nnet ;
   TestNetResults *tnres ;
   ARMA *arma, **armas ;
   Orthog *orthog, **oth ;
   ConfComp *ccptr ;

   if (results != NULL)
      strcpy ( results , "" ) ;

/*
   Act on the command
*/

   if (! id)      // NULL input line or other simple problem
      return 0 ;

   if (id == ID_PRED_INITIALIZE) {
      learn_params.ap = &anneal_params ;
      defaults ( &net_params , &learn_params , &misc_params ) ;
      arma_fixed = 0 ;
      if (strlen ( audit_log )) {
         strcpy ( filename , audit_log ) ;
         _fullpath ( audit_log , filename , 256 ) ;
         if ((fp = fopen ( audit_log , "wt" )) != NULL) {
            fprintf ( fp , "PREDICT audit log (%s)", audit_log ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_AUDIT_LOG) {
      if ((! rest)  ||  ((n=strlen (rest)) == 0))
         strcpy ( audit_log , "" ) ;
      else
         _fullpath ( audit_log , rest , 256 ) ;
      return 0 ;
      }

   if (id == ID_PRED_BYE) {
      MEMTEXT ( "Final BYE cleanup..." ) ;
      clear_all () ;
      return 1 ;
      }

   if (id == ID_PRED_CLEAR_ALL) {
      MEMTEXT ( "Clear all..." ) ;
      clear_all () ;
      defaults ( &net_params , &learn_params , &misc_params ) ;
      arma_fixed = 0 ;
      return 0 ;
      }

   if (id == ID_PRED_PROGRESS_ON) {
      learn_params.progress = 1 ;
      return 0 ;
      }

   if (id == ID_PRED_PROGRESS_OFF) {
      learn_params.progress = 0 ;
      return 0 ;
      }

   if (id == ID_PRED_COMMANDFILE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No file name specified" ) ;
         return -1 ;
         }
      fp = fopen ( rest , "rt" ) ;
      if (fp == NULL) {
         sprintf ( error , "Cannot open %s.", rest ) ;
         return -1 ;
         }

      MEMTEXT ( "PROCESS: Initial cdata" ) ;
      cdata = (char *) MALLOC ( PREDICT_CBLOCK_SIZE ) ;
      if (cdata == NULL) {
         strcpy ( error , "Insufficient memory to read COMMAND FILE" ) ;
         return -1 ;
         }

      istart = 0 ;
      avail = PREDICT_CBLOCK_SIZE ;
      for (;;) {
         if (avail < PREDICT_CBLOCK_SIZE / 4) { // Need more?
            MEMTEXT ( "PROCESS: Expand cdata" ) ;
            cdata = (char *) REALLOC ( cdata , istart + PREDICT_CBLOCK_SIZE ) ;
            if (cdata == NULL) {
               strcpy ( error , "Insufficient memory to read COMMAND FILE" ) ;
               return -1 ;
               }
            avail = PREDICT_CBLOCK_SIZE ;
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
         strcpy ( error , "No data in COMMAND FILE" ) ;
         MEMTEXT ( "PROCESS: Error free cdata" ) ;
         FREE ( cdata ) ;
         return -1 ;
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

   if (id == ID_PRED_NAME) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal names specified" ) ;
         return -1 ;
         }
      if (misc_params.names != NULL) {   // If there are already names here
         MEMTEXT ( "PROCESS: delete previous names" ) ;
         delete ( misc_params.names ) ;  // Get rid of them
         }
      misc_params.names = new Strings ( rest ) ;
      if ((! misc_params.names)  ||  ! misc_params.names->n) {
         if (misc_params.names) {
            delete misc_params.names ;
            misc_params.names = NULL ;
            }
         strcpy ( error , "Insufficient memory" ) ;
         return -1 ;
         }
      if (misc_params.names->is_duplication ()) {
         strcpy ( error , "Cannot duplicate signal names" ) ;
         return -1 ;
         }
      return 0 ;
      }

   if (id == ID_PRED_READSIG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal file specified" ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (readsig ( &misc_params , rest , &n_signals , &signals , error ))
         return -1 ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf( fp, "\nRead signal file '%s' to procure signal(s):", rest);
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SAVESIG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " TO " , 4 ))
            name[i] = rest[i] ;
         else
            break ;
         }
      name[i] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+4))
         strcpy ( filename , rest+i+4 ) ;
      else {
         strcpy ( error , "No 'TO file' name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_signals ; i++) {
         if (! strcmp ( signals[i]->name , name )) {
            ret = sig_save ( signals[i] , filename ) ;
            switch (ret) {
               case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
               case 2: strcpy ( error , "Unable to write file" ) ; return -1 ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nSave signal = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "Signal '%s' does not exist", name ) ;
      return -1 ;
      }

   if ((id == ID_PRED_DISPLAY)  ||  (id == ID_PRED_DISPLAY_CONFIDENCE)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=n_signals-1 ; i>=0 ; i--) {
         if (! strcmp ( rest , signals[i]->name ))
            break ;
         }
      if (i < 0) {
         sprintf ( error , "Signal '%s' does not exist." , rest ) ;
         return -1 ;
         }
      if ((id == ID_PRED_DISPLAY_CONFIDENCE)  &&  ! signals[i]->npred) {
         sprintf ( error , "Signal '%s' has no confidence information.", rest );
         return -1 ;
         }
      if (misc_params.display_domain0 >= (signals[i]->n - 1)) {
         sprintf( error , "Signal '%s' is shorter than the DOMAIN start.",rest);
         return -1 ;
         }
      if ((misc_params.display_domain1 - misc_params.display_domain0) < 2) {
         sprintf( error , "Display domain must be at least 2 points." ) ;
         return -1 ;
         }
      display ( signals[i] , &misc_params , id ) ;
      return 0 ;
      }

   if (id == ID_PRED_DISPLAY_RANGE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No display range specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "OPTIMAL" )) {
         misc_params.display_range = 0 ;
         return 0 ;
         }
      else if (! strcmp ( rest , "SYMMETRIC" )) {
         misc_params.display_range = 1 ;
         return 0 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal DISPLAY RANGE = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.display_min = p ;
      while (((*rest >= '0') && (*rest <= '9'))  ||  (*rest == '.')
             ||  (*rest == '-'))
         ++rest ;  // Skip min
      while (*rest  &&  !(((*rest >= '0') && (*rest <= '9'))  ||  (*rest == '.')
             ||  (*rest == '-')))
         ++rest ;  // Skip to max
      if (*rest) {
         n = sscanf ( rest , "%lf" , &p ) ;
         if (n > 0) {
            if (p > misc_params.display_min) {
               misc_params.display_max = p ;
               misc_params.display_range = 2 ;
               return 0 ;
               }
            }
         }
      sprintf ( error , "Illegal DISPLAY RANGE = %s", rest ) ;
      misc_params.display_range = 0 ;
      return -1 ;
      }

   if (id == ID_PRED_DISPLAY_DOMAIN) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No display domain specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "ALL" )) {
         misc_params.display_domain0 = 0 ;
         misc_params.display_domain1 = MAXPOSNUM ;
         return 0 ;
         }
      n = sscanf ( rest , "%d" , &m ) ;
      if ((n <= 0)  ||  (m < 0)) {
         sprintf ( error , "Illegal DISPLAY DOMAIN = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.display_domain0 = m ;
      while ((*rest >= '0') && (*rest <= '9'))
         ++rest ;  // Skip min
      while (*rest  &&  ! ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip to max
      if (*rest) {
         n = sscanf ( rest , "%d" , &m ) ;
         if ((n > 0)  &&  (m > misc_params.display_domain0)) {
            misc_params.display_domain1 = m ;
            return 0 ;
            }
         }
      sprintf ( error , "Illegal DISPLAY DOMAIN = %s", rest ) ;
      misc_params.display_domain0 = 0 ;
      misc_params.display_domain1 = MAXPOSNUM ;
      return -1 ;
      }

   if (id == ID_PRED_DISPLAY_RATE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No display rate specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal DISPLAY RATE = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.display_rate = p ;
      return 0 ;
      }

   if (id == ID_PRED_DISPLAY_ORIGIN) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No display origin specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal DISPLAY ORIGIN = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.display_origin = p ;
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_INPUT) {
      clear_io ( 1 , &n_ins_outs , &ins_outs ) ;
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_OUTPUT) {
      clear_io ( 0 , &n_ins_outs , &ins_outs ) ;
      return 0 ;
      }

   if (id == ID_PRED_INPUT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      ret = append_io ( 1 , rest , n_signals , signals ,
                        &n_ins_outs , &ins_outs ) ;
      switch (ret) {
         case 1: strcpy ( error , "This signal does not exist" ) ; return -1 ;
         case 2: strcpy ( error , "Insufficient memory" ) ; return -1 ;
         case 3: strcpy ( error , "Min lag exceeds max lag" ) ; return -1 ;
         }
      return 0 ;
      }

   if (id == ID_PRED_OUTPUT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      ret = append_io ( 0 , rest , n_signals , signals ,
                        &n_ins_outs , &ins_outs ) ;
      switch (ret) {
         case 1: strcpy ( error , "This signal does not exist" ) ; return -1 ;
         case 2: strcpy ( error , "Insufficient memory" ) ; return -1 ;
         case 3: strcpy ( error , "Min lag exceeds max lag" ) ; return -1 ;
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_CLASSES) {
      if (n_classes) {
         MEMTEXT ( "PROCESS: classes" ) ;
         for (i=0 ; i<n_classes ; i++) {
            if (classes[i] != NULL)
               FREE ( classes[i] ) ;
            }
         FREE ( classes ) ;
         classes = NULL ;
         n_classes = 0 ;
         if (trnset != NULL) {
            MEMTEXT ( "CLEAR CLASSES trnset" ) ;
            delete trnset ;
            trnset = NULL ;
            }
         if (testset != NULL) {
            MEMTEXT ( "CLEAR CLASSES testset" ) ;
            delete testset ;
            testset = NULL ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLASS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No class name specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "REJECT" )) {
         misc_params.classif_output = 0 ;
         return 0 ;
         }
      for (i=0 ; i<n_classes ; i++) {
         if (! strcmp ( classes[i] , rest ))
            break ;
         }
      if (i == n_classes) {
         MEMTEXT ( "PROCESS: adding a class (2)" ) ;
         if (n_classes)
            cptr = (char **) REALLOC ( classes, (n_classes+1) * sizeof(char *));
         else 
            cptr = (char **) MALLOC ( (n_classes+1) * sizeof(char *) ) ;
         if (cptr == NULL) {
            strcpy ( error , "Insufficient memory" ) ;
            return -1 ;
            }
         classes = cptr ;
         classes[n_classes] = (char *) MALLOC ( strlen ( rest ) + 1 ) ;
         if (classes[n_classes] == NULL) {
            strcpy ( error , "Insufficient memory" ) ;
            return -1 ;
            }
         strcpy ( classes[n_classes] , rest ) ;
         ++n_classes ;
         }
      misc_params.classif_output = i + 1 ;  // Zero reserved for REJECT
      return 0 ;
      }

   if (id == ID_PRED_PRIOR) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No prior specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "N" )) {
         misc_params.classif_prior = -1.0 ;      // Flag to use class count
         return 0 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal PRIOR = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.classif_prior = p ;
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_TRAINING) {
      if (trnset != NULL) {
         MEMTEXT ( "trnset" ) ;
         delete trnset ;
         trnset = NULL ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nClear training set" ) ;
            fclose ( fp ) ;
            }
         }
      }

   if (id == ID_PRED_CLEAR_TEST) {
      if (testset != NULL) {
         MEMTEXT ( "testset" ) ;
         delete testset ;
         testset = NULL ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nClear test set" ) ;
            fclose ( fp ) ;
            }
         }
      }

   if ((id == ID_PRED_CUMULATE_TRAINING)  ||  (id == ID_PRED_CUMULATE_TEST)) {
      if (id == ID_PRED_CUMULATE_TRAINING)
         tset = &trnset ;
      else 
         tset = &testset ;

		n_inputs = n_outputs = 0 ; // Count ins and outs
		for (i=0 ; i<n_ins_outs ; i++) {
			if (ins_outs[i]->is_input)
				n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
			else 
				n_outputs+= ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
			}

		if (! n_inputs) {
			strcpy ( error , "No inputs have been defined" ) ;
			return -1 ;
			}

		if (n_outputs) {
			if (n_classes) {
				strcpy ( error , "Cannot mix classification and prediction" ) ;
				return -1 ;
				}
			else
				out_model = OUTMOD_MAPPING ;
			}
		else {
         out_model = OUTMOD_CLASSIFICATION ;
			n_outputs = n_classes ; // This may be 0 if just REJECTs
			}

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_CUMULATE_TRAINING)
               fprintf ( fp , "\nCumulate training set with inputs/outputs:" ) ;
            else 
               fprintf ( fp , "\nCumulate test set with inputs/outputs:" ) ;
            for (i=0 ; i<n_ins_outs ; i++) {
               if (ins_outs[i]->is_input) {
                  j = ins_outs[i]->which ;
                  fprintf ( fp , "\n  %s", signals[j]->name ) ;
                  if (ins_outs[i]->minlag  ||  ins_outs[i]->maxlag)
                     fprintf ( fp , "  Lag = %d", ins_outs[i]->minlag ) ;
                  if (ins_outs[i]->maxlag > ins_outs[i]->minlag)
                     fprintf ( fp , " to %d", ins_outs[i]->maxlag ) ;
                  fprintf ( fp , " (input)" ) ;
                  }
               }
            if (out_model == OUTMOD_CLASSIFICATION) {
               if (misc_params.classif_output)
                  fprintf ( fp , "\n  Output class = %s",
                            classes[misc_params.classif_output-1] ) ;
               else 
                  fprintf ( fp , "\n  Output class = REJECT" ) ;
               fprintf ( fp , "  Prior = %lf", misc_params.classif_prior ) ;
               }
            else {
               for (i=0 ; i<n_ins_outs ; i++) {
                  if (! ins_outs[i]->is_input) {
                     j = ins_outs[i]->which ;
                     fprintf ( fp , "\n  %s", signals[j]->name ) ;
                     if (ins_outs[i]->minlag || ins_outs[i]->maxlag)
                        fprintf ( fp, "  Lead = %d", ins_outs[i]->minlag);
                     if (ins_outs[i]->maxlag > ins_outs[i]->minlag)
                        fprintf ( fp , " to %d", ins_outs[i]->maxlag ) ;
                     fprintf ( fp , " (output)" ) ;
                     }
                  }
               }
            fclose ( fp ) ;
            }
         }

		if (*tset == NULL) {
			MEMTEXT ( "PROCESS: new tset" ) ;
			*tset = new TrainingSet ( out_model , n_inputs , n_outputs ,
                                   n_ins_outs , ins_outs ) ;
			if (*tset == NULL) {
				strcpy ( error , "Insufficient memory to create training/test set");
				return -1 ;
				}
			}

      net_params.n_inputs = n_inputs ;
      net_params.n_outputs = n_outputs ;
      net_params.out_model = out_model ;
		ret = (*tset)->train ( &net_params , &misc_params ,
                             n_ins_outs , ins_outs , signals ) ;
		switch (ret) {
			case 1: strcpy ( error ,
                          "Insufficient memory to cumulate training/test set" ) ;
					  return -1 ;
			case 2: strcpy ( error , "Cannot append incompatible data" ) ;
					  return -1 ;
			case 3: strcpy ( error , "There are no valid cases to append" ) ;
					  return -1 ;
			}

		return 0 ;
		}

   if (id == ID_PRED_CUMULATE_INCLUDE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No include quantity specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &m ) ;
      if ((m < 0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal CUMULATE INCLUDE = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.include = m ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nCumulate include = %d", m ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CUMULATE_EXCLUDE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No exclude quantity specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &m ) ;
      if ((m < 0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal CUMULATE EXCLUDE = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.exclude = m ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nCumulate exclude = %d", m ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_NETMOD) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network model specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "PNN" ))
         n = NETMOD_PNN ;
      else if (! strcmp ( rest , "SEPVAR" ))
         n = NETMOD_SEPVAR ;
      else if (! strcmp ( rest , "SEPCLASS" ))
         n = NETMOD_SEPCLASS ;
      else if (! strcmp ( rest , "MLFN" ))
         n = NETMOD_MLFN ;
      else {
         sprintf ( error , "Illegal NETWORK MODEL = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nNetwork model = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      if (net_params.net_model == n)
         return 0 ;
      net_params.net_model = n ;
      return 0 ;
      }

   if (id == ID_PRED_KERNEL) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No kernel specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "GAUSS" ))
         n = KERNEL_GAUSS ;
      else if (! strcmp ( rest , "RECIPROCAL" ))
         n = KERNEL_RECIP ;
      else {
         sprintf ( error , "Illegal KERNEL = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nKernel = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      if (net_params.kernel == n)
         return 0 ;
      net_params.kernel = n ;
      return 0 ;
      }

   if (id == ID_PRED_MLFN_HID1) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No hidden neurons specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal MLFN HID 1 = %s", rest ) ;
         return -1 ;
         }
      else
         net_params.n_hidden1 = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN Hid1 = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_HID2) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No hidden neurons specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal MLFN HID 2 = %s", rest ) ;
         return -1 ;
         }
      else
         net_params.n_hidden2 = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN Hid2 = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_OUTPUT_ACT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No output activation specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "LINEAR" ))
         net_params.linear = 1 ;
      else if (! strcmp ( rest , "NONLINEAR" ))
         net_params.linear = 0 ;
      else {
         sprintf ( error , "Illegal MLFN OUTPUT ACTIVATION = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN output activation = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_DOMAIN) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No output domain specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "REAL" ))
         net_params.domain = DOMAIN_REAL ;
      else if (! strcmp ( rest , "COMPLEX" ))
         net_params.domain = DOMAIN_COMPLEX ;
      else if (! strcmp ( rest , "COMPLEX INPUT" ))
         net_params.domain = DOMAIN_COMPLEX_INPUT ;
      else if (! strcmp ( rest , "COMPLEX HIDDEN" ))
         net_params.domain = DOMAIN_COMPLEX_HIDDEN ;
      else {
         sprintf ( error , "Illegal MLFN DOMAIN = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN domain = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_QUITERR) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No quitting error specified" ) ;
         return -1 ;
         }
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( error , "Illegal ALLOWABLE ERROR = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.quit_err = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nAllowable error = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SIGLO) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No sigma limit specified" ) ;
         return -1 ;
         }
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( error , "Illegal SIGMA LOW = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.siglo = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nSigma low = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SIGHI) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No sigma limit specified" ) ;
         return -1 ;
         }
      p = atof ( rest ) ;
      if (p < 0.0) {
         sprintf ( error , "Illegal SIGMA HIGH = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.sighi = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nSigma high = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SIGTRIES) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No sigma tries specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 3) {
         sprintf ( error , "Illegal SIGMA TRIES = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.nsigs = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nSigma tries = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_ERROR_TYPE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No error type specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "MEAN SQUARE" ))
         learn_params.errtype = ERRTYPE_MSE ;
      else if (! strcmp ( rest , "ABSOLUTE" ))
         learn_params.errtype = ERRTYPE_ABS ;
      else if (! strcmp ( rest , "KK" ))
         learn_params.errtype = ERRTYPE_KK ;
      else if (! strcmp ( rest , "CROSS ENTROPY" ))
         learn_params.errtype = ERRTYPE_XENT ;
      else {
         sprintf ( error , "Illegal MLFN ERROR TYPE = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN error type = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_LEARNING) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No MLFN learning method specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "AN1" ))
         learn_params.method = METHOD_AN1 ;
      else if (! strcmp ( rest , "AN2" ))
         learn_params.method = METHOD_AN2 ;
      else if (! strcmp ( rest , "ANNEALING_CJ" ))
         learn_params.method = METHOD_AN1_CJ ;
      else if (! strcmp ( rest , "AN1_CJ" ))
         learn_params.method = METHOD_AN1_CJ ;
      else if (! strcmp ( rest , "AN2_CJ" ))
         learn_params.method = METHOD_AN2_CJ ;
      else if (! strcmp ( rest , "AN1_LM" ))
         learn_params.method = METHOD_AN1_LM ;
      else if (! strcmp ( rest , "AN2_LM" ))
         learn_params.method = METHOD_AN2_LM ;
      else if (! strcmp ( rest , "REGRESS_CJ" ))
         learn_params.method = METHOD_REGRS_CJ ;
      else if (! strcmp ( rest , "REGRESS_LM" ))
         learn_params.method = METHOD_REGRS_LM ;
      else if (! strcmp ( rest , "SS" ))
         learn_params.method = METHOD_SS ;
      else if (! strcmp ( rest , "SSG" ))
         learn_params.method = METHOD_SSG ;
      else {
         sprintf ( error , "Illegal MLFN LEARNING METHOD = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN learning method = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_RESTARTS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No MLFN restarts specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal MLFN RESTARTS = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.retries = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN restarts = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MLFN_PRETRIES) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No MLFN pretries specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal MLFN PRETRIES = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.pretries = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN pretries = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ACCURACY) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No ACCURACY specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 1) {
         sprintf ( error , "Illegal ACCURACY = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.acc = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nAccuracy = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_REFINE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No REFINE specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal REFINE = %s", rest ) ;
         return -1 ;
         }
      else
         learn_params.refine = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nRefine = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_TEMPS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.ntempI = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init temps = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_TEMPS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.ntempE = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape temps = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_ITERS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.niterI = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init iters = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_ITERS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n <= 0) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.niterE = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape iters = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_SETBACK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.sbI = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init setback = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_SETBACK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = atoi ( rest ) ;
      if (n < 0) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.sbE = n ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape setback = %d", n ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_START) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.startI = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init start temp = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_START) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.startE = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape start temp = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_STOP) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.endI = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init stop temp = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_STOP) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.endE = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape stop temp = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_RATIO) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.ratioI = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init ratio = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_RATIO) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ANNEALING parameter = %s", rest ) ;
         return -1 ;
         }
      else
         anneal_params.ratioE = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape ratio = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_REDUCTION) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "EXPONENTIAL" ))
         anneal_params.reducI = ANNEAL_REDUCE_EXPONENTIAL ;
      else if (! strcmp ( rest , "FAST" ))
         anneal_params.reducI = ANNEAL_REDUCE_FAST ;
      else {
         sprintf ( error , "Illegal ANNEALING REDUCTION = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init reduction = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_REDUCTION) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "EXPONENTIAL" ))
         anneal_params.reducE = ANNEAL_REDUCE_EXPONENTIAL ;
      else if (! strcmp ( rest , "FAST" ))
         anneal_params.reducE = ANNEAL_REDUCE_FAST ;
      else {
         sprintf ( error , "Illegal ANNEALING REDUCTION = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape reduction = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_RANDOM) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "GAUSS" ))
         anneal_params.randomI = NormalDensity ;
      else if (! strcmp ( rest , "CAUCHY" ))
         anneal_params.randomI = CauchyDensity ;
      else {
         sprintf ( error , "Illegal ANNEALING RANDOM = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init random = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_RANDOM) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "GAUSS" ))
         anneal_params.randomE = NormalDensity ;
      else if (! strcmp ( rest , "CAUCHY" ))
         anneal_params.randomE = CauchyDensity ;
      else {
         sprintf ( error , "Illegal ANNEALING RANDOM = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape random = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_INIT_ACCEPT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "CLIMB" ))
         anneal_params.climbI = 1 ;
      else if (! strcmp ( rest , "NOCLIMB" ))
         anneal_params.climbI = 0 ;
      else {
         sprintf ( error , "Illegal ANNEALING ACCEPT = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal init accept = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ANNEAL_ESCAPE_ACCEPT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No annealing parameter specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "CLIMB" ))
         anneal_params.climbE = 1 ;
      else if (! strcmp ( rest , "NOCLIMB" ))
         anneal_params.climbE = 0 ;
      else {
         sprintf ( error , "Illegal ANNEALING ACCEPT = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nMLFN anneal escape accept = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_NETWORKS) {
      if (n_networks) {
         MEMTEXT ( "PROCESS: networks" ) ;
         for (i=0 ; i<n_networks ; i++) {
            if (networks[i] != NULL)
               delete networks[i] ;
            }
         FREE ( networks ) ;
         networks = NULL ;
         n_networks = 0 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nClear networks" ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_NETWORK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , rest )) {
            MEMTEXT ( "PROCESS: network" ) ;
            delete networks[i] ;
            for (j=i+1 ; j<n_networks ; j++)
               networks[j-1] = networks[j] ;
            --n_networks ;
            if (n_networks)
               networks = (Network **) REALLOC ( networks ,
                                              n_networks * sizeof(Network *)) ;
            else {
               MEMTEXT ( "PROCESS: network array" ) ;
               FREE ( networks ) ;
               networks = NULL ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nClear network = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "Network '%s' does not exist", rest ) ;
      return -1 ;
      }

   if ((id == ID_PRED_TRAIN_NETWORK)  ||  (id == ID_PRED_CVTRAIN_NETWORK)) {
      if ((! rest)  ||  ((n=strlen (rest)) == 0)) {
         strcpy ( error , "No network name specified" ) ;
         return -1 ;
         }
      if (check_training_set ( error ))
         return -1 ;
      if (alphabetize ()) {  // Put class names in alphabetical order
         strcpy ( error , "Insufficient memory" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_networks ; i++) {    // Is this named network here?
         if (! strcmp ( networks[i]->name , rest )) {
            net = networks[i] ;
            if (check_set_net_compatibility ( trnset , net , error ))
               return -1 ;
            net_params.net_model = net->model ;
            net_params.out_model = net->output_mode ;
            net_params.n_inputs = net->n_inputs ;
            net_params.n_outputs = net->n_outputs ;
            if ((net_params.net_model == NETMOD_PNN)  ||
                 (net_params.net_model == NETMOD_SEPVAR) ||
                 (net_params.net_model == NETMOD_SEPCLASS))
               net_params.kernel = ((PNNet *)net)->kernel ;
            if (net_params.net_model == NETMOD_MLFN) {
               net_params.domain = ((MLFN *) net)->domain ;
               net_params.linear = ((MLFN *) net)->outlin ;
               net_params.n_hidden1 = ((MLFN *) net)->nhid1 ;
               net_params.n_hidden2 = ((MLFN *) net)->nhid2 ;
               }
            break ;
            }
         }
      if (i == n_networks) {   // Not here, so create a new one
         net_params.out_model = trnset->output_mode ;
         net_params.n_inputs = trnset->n_inputs ;
         net_params.n_outputs = trnset->n_outputs ;
         net_params.classnames = classes ;
         if (check_net_params ( error ))
            return -1 ;
         if (check_learn_params ( error ))
            return -1 ;
         MEMTEXT ( "PROCESS: new Network" ) ;
         if (net_params.net_model == NETMOD_PNN)
            net = new PNNbasic ( rest , &net_params ) ;
         else if (net_params.net_model == NETMOD_SEPVAR)
            net = new PNNsepvar ( rest , &net_params ) ;
         else if (net_params.net_model == NETMOD_SEPCLASS)
            net = new PNNsepclass ( rest , &net_params ) ;
         else if (net_params.net_model == NETMOD_MLFN)
            net = new MLFN ( rest , &net_params ) ;
         if ((net == NULL)  ||  (! net->ok)) {  // Malloc failure?
            strcpy ( error , "Insufficient memory to create network." ) ;
            if (net != NULL)
               delete net ;
            return -1 ;
            }
         MEMTEXT ( "PROCESS: network array" ) ;
         if (networks)
            nnet = (Network **) REALLOC ( networks ,
                                       (n_networks+1) * sizeof(Network *) ) ;
         else 
            nnet = (Network **) MALLOC ( (n_networks+1) * sizeof(Network *) ) ;
         if (nnet == NULL) {
            strcpy ( error , "Insufficient memory to append network." ) ;
            delete net ;
            return -1 ;
            }
         networks = nnet ;
         networks[n_networks++] = net ;
         }
      if (check_learn_params ( error ))
         return -1 ;
      if (id == ID_PRED_TRAIN_NETWORK)
         ret = net->learn ( trnset , &learn_params ) ;
      else if (id == ID_PRED_CVTRAIN_NETWORK)
         ret = cvtrain ( trnset , net , &net_params , &learn_params , &p ) ;
      if (ret < 0) {
         strcpy ( error , "Insufficient memory to train network." ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nTrain network = %s", rest ) ;
            if (net->errtype) {
               fprintf ( fp , "  Training set error = %lf", net->neterr ) ;
               if ((id == ID_PRED_CVTRAIN_NETWORK)  &&  (p >= 0.0))
                  fprintf ( fp , "  (CV = %lf)", p ) ;
               if (ret > 0)
                  fprintf ( fp , " after user interruption" ) ;
               }
            else 
               fprintf ( fp , "Interrupted by user before weights found" ) ;
            fclose ( fp ) ;
            }
         }
      if (net->errtype) {
         sprintf ( results , "Training set error = %lf", net->neterr ) ;
         if ((id == ID_PRED_CVTRAIN_NETWORK)  &&  (p >= 0.0)) {
            sprintf ( msg , "  (CV = %lf)", p ) ;
            strcat ( results , msg ) ;
            }
         }
      else 
         strcpy ( results , "Interrupted by user before weights found" ) ;
      return 0 ;
      }

   if (id == ID_PRED_SAVE_NETWORK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " TO " , 4 ))
            netname[i] = rest[i] ;
         else
            break ;
         }
      j = i ;
      while (netname[j-1] == ' ')  // Remove trailing blanks from name
         --j ;
      netname[j] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+4))
         strcpy ( filename , rest+i+4 ) ;
      else {
         strcpy ( error , "No 'TO file' name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , netname )) {
            if (! networks[i]->errtype) {
               strcpy ( error , "This network is not trained" ) ;
               return -1 ;
               }
            ret = wt_save ( networks[i] , filename ) ;
            switch (ret) {
               case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
               case 2: strcpy ( error , "Unable to write file" ) ; return -1 ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nSave network = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "Network '%s' does not exist", netname ) ;
      return -1 ;
      }

   if (id == ID_PRED_RESTORE_NETWORK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No net/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " FROM " , 6 ))
            netname[i] = rest[i] ;
         else
            break ;
         }
      j = i ;
      while (netname[j-1] == ' ')  // Remove trailing blanks from name
         --j ;
      netname[j] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+6))
         strcpy ( filename , rest+i+6 ) ;
      else {
         strcpy ( error , "No 'FROM file' name specified" ) ;
         return -1 ;
         }
      net = wt_restore ( netname , filename , &ret ) ;
      if (net == NULL) {
         switch (ret) {
            case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
            case 2: strcpy ( error , "Unable to read file" ) ; return -1 ;
            case 3: strcpy ( error , "Illegal file header" ) ; return -1 ;
            case 4: strcpy ( error , "Insufficient memory" ) ; return -1 ;
            }
         }
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , netname )) {
            MEMTEXT ( "PROCESS: network deleted for replacement" ) ;
            delete networks[i] ;
            networks[i] = net ;
            break ;
            }
         }
      if (i == n_networks) {
         MEMTEXT ( "PROCESS: network array" ) ;
         if (n_networks++)
            nnet = (Network **) REALLOC ( networks ,
                                          n_networks * sizeof(Network *)) ;
         else
            nnet = (Network **) MALLOC ( n_networks * sizeof(Network *)) ;
         if (nnet == NULL) {
            strcpy ( error , "Insufficient memory" ) ;
            delete net ;
            return -1 ;
            }
         networks = nnet ;
         networks[n_networks-1] = net ;
         }
      net_params.net_model = net->model ;
      net_params.out_model = net->output_mode ;
      net_params.n_inputs = net->n_inputs ;
      net_params.n_outputs = net->n_outputs ;
      if ((net_params.net_model == NETMOD_PNN)  ||
           (net_params.net_model == NETMOD_SEPVAR) ||
           (net_params.net_model == NETMOD_SEPCLASS))
         net_params.kernel = ((PNNet *)net)->kernel ;
      if (net_params.net_model == NETMOD_MLFN) {
         net_params.domain = ((MLFN *) net)->domain ;
         net_params.linear = ((MLFN *) net)->outlin ;
         net_params.n_hidden1 = ((MLFN *) net)->nhid1 ;
         net_params.n_hidden2 = ((MLFN *) net)->nhid2 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nRestore network = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_PRINT_NETWORK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " TO " , 4 ))
            netname[i] = rest[i] ;
         else
            break ;
         }
      j = i ;
      while (netname[j-1] == ' ')  // Remove trailing blanks from name
         --j ;
      netname[j] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+4))
         strcpy ( filename , rest+i+4 ) ;
      else {
         strcpy ( error , "No 'TO file' name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , netname )) {
            if (! networks[i]->errtype) {
               strcpy ( error , "This network is not trained" ) ;
               return -1 ;
               }
            ret = networks[i]->wt_print ( filename ) ;
            switch (ret) {
               case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
               case 2: strcpy ( error , "Unable to write file" ) ; return -1 ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nPrint network = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "Network '%s' does not exist", netname ) ;
      return -1 ;
      }

   if (id == ID_PRED_THRESHOLD) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No threshold specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p < 0.0)  ||  (p > 100.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal THRESHOLD = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.threshold = p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nThreshold = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if ((id == ID_PRED_TEST_NETWORK)  ||  (id == ID_PRED_EXTENDED_TEST_NETWORK)){
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , rest )) {
            net = networks[i] ;
            break ;
            }
         }
      if (i == n_networks) {
         sprintf ( error , "Network '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (! net->errtype) {
         strcpy ( error , "This network is not trained" ) ;
         return -1 ;
         }
      if ((testset == NULL)  ||  (testset->ntrain == 0)) {
         strcpy ( error , "Cannot TEST NETWORK; No test set exists." ) ;
         return -1 ;
         }
      if (testset->ntrain < 2) {
         strcpy ( error , "Cannot TEST NETWORK; Need at least 2 cases." ) ;
         return -1 ;
         }
      if (check_test_set ( error ))
         return -1 ;
      if (alphabetize ()) {  // Put class names in alphabetical order
         strcpy ( error , "Insufficient memory" ) ;
         return -1 ;
         }
      if (check_set_net_compatibility ( testset , net , error ))
         return -1 ;
      MEMTEXT ( "PROCESS: testnet results" ) ;
      if ((tnres = (TestNetResults *) MALLOC
                   ( testset->n_outputs * sizeof(TestNetResults))) == NULL) {
         strcpy ( error , "Insufficient memory to test network" ) ;
         return -1 ;
         }
      if (net->output_mode == OUTMOD_CLASSIFICATION) {
         MEMTEXT ( "PROCESS: test confusion" ) ;
         confusion = (int *) MALLOC ( (testset->n_outputs+1) * (testset->n_outputs+1) *
                                      sizeof(int) ) ;
         if (confusion == NULL) {
            FREE ( tnres ) ;
            strcpy ( error , "Insufficient memory to test network" ) ;
            return -1 ;
            }
         }
      else
         confusion = NULL ;
      ret = net->testnet ( testset , misc_params.threshold / 100.0 , confusion , &confuse ,
                           (id == ID_PRED_EXTENDED_TEST_NETWORK) , tnres ) ;
      if (ret < 0) {
         strcpy ( error , "Insufficient memory to test network" ) ;
         return -1 ;
         }
      else if (ret > 0) {
         strcpy ( error , "User interrupted" ) ;
         return -1 ;
         }

      p = 0.0 ;
      for (i=0 ; i<testset->n_outputs ; i++)
         p += tnres[i].mse ;
      sprintf ( results , "Mean squared error = %lf" , p / testset->n_outputs ) ;

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nTest network = %s  MSE = %lf",
                      rest, p / testset->n_outputs ) ;
            for (i=0 ; i<testset->n_outputs ; i++) {
               if (net->output_mode == OUTMOD_CLASSIFICATION)
                  fprintf ( fp , "\n%s:", classes[i] ) ;
               else 
                  fprintf ( fp , "\nOutput %d:", i+1 ) ;
               fprintf ( fp , " MSE=%lf (%lf)  MAE=%lf (%lf)",
                         tnres[i].mse, tnres[i].stderr_mse,
                         tnres[i].mae, tnres[i].stderr_mae ) ;
               fprintf ( fp ,
         "\n  p5=%.4lf  p10=%.4lf  p25=%.4lf  p75=%.4lf  p90=%.4lf  p95=%.4lf",
                  tnres[i].p5, tnres[i].p10, tnres[i].p25,
                  tnres[i].p75, tnres[i].p90, tnres[i].p95 ) ;
               if (id == ID_PRED_EXTENDED_TEST_NETWORK) {
                  fprintf ( fp , "\n  90'th percentile absolute error = %lf",
                            tnres[i].abs90 ) ;
                  fprintf(fp,"\n  Jackknife bias=%lf  unbiased=%lf  stderr=%lf",
                          tnres[i].jackbias, tnres[i].abs90-tnres[i].jackbias,
                          tnres[i].jackstd ) ;
                  fprintf(fp,"\n  Bootstrap bias=%lf  unbiased=%lf  stderr=%lf",
                          tnres[i].bootbias, tnres[i].abs90-tnres[i].bootbias,
                          tnres[i].bootstd ) ;
                  }
               }
            if (net->output_mode == OUTMOD_CLASSIFICATION) {
               fprintf( fp ,
                     "\nConfusion matrix (Total misclassified = %lf percent)",
                     100.0 * confuse ) ;
               m = testset->ntrain ;              // Total number of cases
               for (i=0 ; i<testset->n_outputs ; i++)  // Minus n in each class
                  m -= testset->nper[i] ;         // Tells us if any rejects
               for (i=0 ; i<=testset->n_outputs ; i++) { // All classes plus reject
                  if (i == testset->n_outputs) {       // If this is the reject row
                     if (m)                       // If there are reject cases
                        fprintf ( fp , "\n         REJECT:" ) ;
                     else
                        break ;
                     }
                  else
                     fprintf ( fp , "\n%15s:", classes[i] ) ;
                  for (j=0 ; j<=testset->n_outputs ; j++)
                     fprintf ( fp , " %4d", confusion[i*(testset->n_outputs+1)+j] ) ;
                  }
               } // If CLASSIFICATION
            fclose ( fp ) ;
            }
         }
      MEMTEXT ( "PROCESS: testnet results" ) ;
      FREE ( tnres ) ;
      if (confusion != NULL) {
         MEMTEXT ( "PROCESS: test confusion" ) ;
         FREE ( confusion ) ;
         }
      return 0 ;
      }

   if (id == ID_PRED_GENERATE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No parameters specified" ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (generate ( &misc_params , rest , &n_signals , &signals , error ))
         return -1 ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf( fp, "\nGenerate = %s to procure signal(s):", rest);
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_NETWORK_PREDICT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &ncases ) ;
      if ((! n)  ||  (ncases < 1)) {
         strcpy ( error , "Illegal predicted signal length specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip number of cases
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , rest )) {
            net = networks[i] ;
            break ;
            }
         }
      if (i == n_networks) {
         sprintf ( error , "Network '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (! net->errtype) {
         strcpy ( error , "This network is not trained" ) ;
         return -1 ;
         }

		n_inputs = 0 ;
		for (i=0 ; i<n_ins_outs ; i++) {
			if (ins_outs[i]->is_input)
				n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
			}

		if (! n_inputs) {
			strcpy ( error , "No inputs have been defined" ) ;
			return -1 ;
			}

      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nPREDICT WITH NETWORK = %s", rest ) ;
            for (i=0 ; i<n_ins_outs ; i++) {
               if (ins_outs[i]->is_input) {
                  j = ins_outs[i]->which ;
                  fprintf ( fp , "\n  %s", signals[j]->name ) ;
                  if (ins_outs[i]->minlag  ||  ins_outs[i]->maxlag)
                     fprintf ( fp , "  Lag = %d", ins_outs[i]->minlag ) ;
                  if (ins_outs[i]->maxlag > ins_outs[i]->minlag)
                     fprintf ( fp , " to %d", ins_outs[i]->maxlag ) ;
                  fprintf ( fp , " (input)" ) ;
                  }
               }
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s (predicted signal)",
                            misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }

      m = net->n_inputs ;
      if ((net->model == NETMOD_MLFN)  &&  (((MLFN *)net)->domain != DOMAIN_REAL))
         m *= 2 ;
   
      if (m != n_inputs) {
         sprintf( error , "Signal inputs (%d) not equal to net inputs (%d)",
                  n_inputs, m ) ;
         return -1 ;
         }

      m = net->n_outputs ;
      if ((net->model == NETMOD_MLFN) && (((MLFN *)net)->domain == DOMAIN_COMPLEX))
         m *= 2 ;
      if (m < misc_params.names->n) {
         sprintf( error , "Net has %d outputs, but %d names specified.",
                  m, misc_params.names->n ) ;
         return -1 ;
         }

      ret = net_pred ( n_inputs , ncases , net , &misc_params ,
                       n_ins_outs , ins_outs ,
                       &n_signals , &signals ) ;

      if (ret < 0) {
         strcpy ( error , "Insufficient memory to predict with network" ) ;
         return -1 ;
         }
      else if (ret > 0) {
         strcpy ( error , "User interrupted" ) ;
         return -1 ;
         }
      return 0 ;
      }

   if (id == ID_PRED_NETWORK_CONFIDENCE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No network name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &ncases ) ;
      if ((! n)  ||  (ncases < 1)) {
         strcpy ( error , "Illegal prediction confidence length specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip number of predictions
      for (i=0 ; i<n_networks ; i++) {
         if (! strcmp ( networks[i]->name , rest )) {
            net = networks[i] ;
            break ;
            }
         }
      if (i == n_networks) {
         sprintf ( error , "Network '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (! net->errtype) {
         strcpy ( error , "This network is not trained" ) ;
         return -1 ;
         }
      if (net->output_mode == OUTMOD_CLASSIFICATION) {
         strcpy ( error , "CONFIDENCE cannot be done in CLASSIFICATION mode." );
         return -1 ;
         }

		n_inputs = 0 ;
		for (i=0 ; i<n_ins_outs ; i++) {
			if (ins_outs[i]->is_input)
				n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
			}

		if (! n_inputs) {
			strcpy ( error , "No inputs have been defined" ) ;
			return -1 ;
			}

      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No output signal name specified" ) ;
         return -1 ;
         }

      for (i=0 ; i<misc_params.names->n ; i++) {
         if (! misc_params.names->len[i])
            continue ;
         for (j=n_signals-1 ; j>=0 ; j--) {
            if (! strcmp ( misc_params.names->start[i] , signals[j]->name ))
               break ;
            }
         if (j < 0) {
            sprintf ( error , "Signal '%s' does not exist." ,
                      misc_params.names->start[i] ) ;
            return -1 ;
            }
         }

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nNETWORK CONFIDENCE = %s", rest ) ;
            for (i=0 ; i<n_ins_outs ; i++) {
               if (ins_outs[i]->is_input) {
                  j = ins_outs[i]->which ;
                  fprintf ( fp , "\n  %s", signals[j]->name ) ;
                  if (ins_outs[i]->minlag  ||  ins_outs[i]->maxlag)
                     fprintf ( fp , "  Lag = %d", ins_outs[i]->minlag ) ;
                  if (ins_outs[i]->maxlag > ins_outs[i]->minlag)
                     fprintf ( fp , " to %d", ins_outs[i]->maxlag ) ;
                  fprintf ( fp , " (input)" ) ;
                  }
               }
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s (predicted signal)",
                            misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }

      m = net->n_inputs ;
      if ((net->model == NETMOD_MLFN)  &&  (((MLFN *)net)->domain != DOMAIN_REAL))
         m *= 2 ;
   
      if (m != n_inputs) {
         sprintf( error , "Signal inputs (%d) not equal to net inputs (%d)",
                  n_inputs, m ) ;
         return -1 ;
         }

      m = net->n_outputs ;
      if ((net->model == NETMOD_MLFN) && (((MLFN *)net)->domain == DOMAIN_COMPLEX))
         m *= 2 ;
      if (m < misc_params.names->n) {
         sprintf( error , "Net has %d outputs, but %d names specified.",
                  m, misc_params.names->n ) ;
         return -1 ;
         }

      ret = net_conf ( n_inputs , ncases , net , &misc_params ,
                       n_ins_outs , ins_outs , n_signals , signals ,
                       n_conf_comps , conf_comps , &p , &t ) ;

      if (ret < 0) {
         strcpy ( error , "Insufficient memory to predict with network" ) ;
         return -1 ;
         }
      else if (ret == 1) {
         strcpy ( error , "User interrupted" ) ;
         return -1 ;
         }
      else if (ret == 2) {
         strcpy ( error , "Too few cases for reliable results." ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp ,
            "\n  There is a 5%% chance each tail probability exceeds %.2lf %%",
               p * 100.0 ) ;
            fprintf ( fp ,
            "\n  There is a %.2lf%% chance the interval exceeds %.2lf %%",
               t * 100.0 , 100.0 * (1.0 - 2.0 * (1.0 - misc_params.conf_prob)));
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_ARMAS) {
      if (n_ARMAs) {
         MEMTEXT ( "PROCESS: ARMAs" ) ;
         for (i=0 ; i<n_ARMAs ; i++) {
            if (ARMAs[i] != NULL)
               delete ARMAs[i] ;
            }
         FREE ( ARMAs ) ;
         ARMAs = NULL ;
         n_ARMAs = 0 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nClear ARMAs" ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_ARMA) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No ARMA name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_ARMAs ; i++) {
         if (! strcmp ( ARMAs[i]->name , rest )) {
            MEMTEXT ( "PROCESS: ARMA" ) ;
            delete ARMAs[i] ;
            for (j=i+1 ; j<n_ARMAs ; j++)
               ARMAs[j-1] = ARMAs[j] ;
            --n_ARMAs ;
            if (n_ARMAs)
               ARMAs = (ARMA **) REALLOC ( ARMAs ,
                                              n_ARMAs * sizeof(ARMA *)) ;
            else {
               MEMTEXT ( "PROCESS: ARMA array" ) ;
               FREE ( ARMAs ) ;
               ARMAs = NULL ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nClear ARMA = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "ARMA '%s' does not exist", rest ) ;
      return -1 ;
      }

   if (id == ID_PRED_TRAIN_ARMA) {
      if ((! rest)  ||  ((n=strlen (rest)) == 0)) {
         strcpy ( error , "No ARMA name specified" ) ;
         return -1 ;
         }
		n_inputs = n_ma = n_outputs = 0 ; // Count inputs, MAs, and outputs
		for (i=0 ; i<n_ins_outs ; i++) {  // Check all ins and outs
			if (ins_outs[i]->is_input) {   // If this is an input
            if (! ins_outs[i]->minlag) {
               strcpy ( error , "Cannot have lag 0 for an ARMA input" ) ;
               return -1 ;
               }
				n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
            }
			else {                         // This is an output
            if (ins_outs[i]->maxlag  &&  ! ins_outs[i]->minlag) {
               strcpy ( error , "Cannot have lag 0 MA term" ) ;
               return -1 ;
               }
				n_ma+= ins_outs[i]->maxlag - ins_outs[i]->minlag ;
            if (ins_outs[i]->minlag)
               ++n_ma ;  // Only lags are MA terms
            for (j=0 ; j<i ; j++) {     // Have we seen this output before?
               if ((! ins_outs[j]->is_input)  &&
                   (ins_outs[i]->which == ins_outs[j]->which))
                  break ;
               }
            if (i == j)       // If this is the first appearance of this output
               ++n_outputs ;  // Count it
            }
			}

		if (! (n_inputs + n_ma)) {
			strcpy ( error , "No inputs (or MAs) have been defined" ) ;
			return -1 ;
			}

		if (! n_outputs) {
			strcpy ( error , "No outputs have been defined" ) ;
         return -1 ;
			}

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf(fp, "\nTrain ARMA=%s with %d inputs, %d MA, and %d outputs",
                      rest, n_inputs, n_ma, n_outputs ) ;
            fclose ( fp ) ;
            }
         }
      for (i=0 ; i<n_ARMAs ; i++) {    // Is this named ARMA here?
         if (! strcmp ( ARMAs[i]->name , rest )) {
            arma = ARMAs[i] ;
            if (check_arma_compatibility ( n_inputs + n_ma , n_outputs ,
                                           arma_fixed , arma , error ))
               return -1 ;
            break ;
            }
         }
      if (i == n_ARMAs) {   // Not here, so create a new one
         MEMTEXT ( "PROCESS: new ARMA" ) ;
         arma = new ARMA ( rest , n_inputs , n_ma , n_outputs ,
                           n_ins_outs , ins_outs , arma_fixed ) ;
         if ((arma == NULL)  ||  (! arma->ok)) {  // Malloc failure?
            strcpy ( error , "Insufficient memory to create ARMA." ) ;
            if (arma != NULL)
               delete arma ;
            return -1 ;
            }
         MEMTEXT ( "PROCESS: ARMA array" ) ;
         if (ARMAs)
            armas = (ARMA **) REALLOC ( ARMAs , (n_ARMAs+1) * sizeof(ARMA *) ) ;
         else 
            armas = (ARMA **) MALLOC ( (n_ARMAs+1) * sizeof(ARMA *) ) ;
         if (armas == NULL) {
            strcpy ( error , "Insufficient memory to append ARMA." ) ;
            delete arma ;
            return -1 ;
            }
         ARMAs = armas ;
         ARMAs[n_ARMAs++] = arma ;
         }
      ret = arma->learn ( n_ins_outs , ins_outs , signals , &learn_params ) ;
      if (ret == -1) {
         strcpy ( error , "Insufficient memory to train ARMA." ) ;
         return -1 ;
         }
      else if (ret == -2) {
         strcpy ( error , "Too few cases to train ARMA." ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (arma->trained) {
               fprintf ( fp , "\n Pooled RMS error = %lf", sqrt(arma->error) ) ;
               if (ret > 0)
                  fprintf ( fp , " after user interruption" ) ;
               m = 0 ;    // Will count unique outputs
               for (iout=0 ; iout<n_ins_outs ; iout++) {  // Do all output sigs
                  if (ins_outs[iout]->is_input)
                     continue ;
                  for (j=0 ; j<iout ; j++) { // Have we seen this output before?
                     if ((! ins_outs[j]->is_input)  &&
                         (ins_outs[iout]->which == ins_outs[j]->which))
                        break ;
                     }
                  if (j < iout)      // If this output has been done already
                     continue ;      // Skip it
                  wptr = arma->wts + m * arma->nvars ;// AR, MA, const
                  fprintf ( fp, "\n Output %s  (noise var=%lf  std=%lf)",
                     signals[ins_outs[iout]->which]->name,
                     arma->var[m], sqrt(arma->var[m]) ) ;
                  ++m ;
                  for (i=0 ; i<n_ins_outs ; i++) {  // Do all weights
                     if (ins_outs[i]->is_input) {
                        for (j=ins_outs[i]->minlag; j<=ins_outs[i]->maxlag; j++)
                           fprintf ( fp ,
                            "\n  %s (input, length=%d)  Lag = %d  weight = %lf",
                            signals[ins_outs[i]->which]->name ,
                            signals[ins_outs[i]->which]->n , j, *wptr++ ) ;
                        } // If this is an input
                     else { // This is an output (MA if lagged)
                        for(j=ins_outs[i]->minlag; j<=ins_outs[i]->maxlag; j++){
                           if (j == 0)    // Only lags
                              continue ;  // are MA terms
                           fprintf ( fp ,
                              "\n  %s (MA, length=%d)  Lag = %d  weight = %lf",
                              signals[ins_outs[i]->which]->name ,
                              signals[ins_outs[i]->which]->n , j, *wptr++ ) ;
                           }
                        } // Else this is an output (MA if lagged)
                     } // For all weights of this output signal
                  if (arma_fixed)
                     fprintf ( fp , "\n  constant fixed at 0" ) ;
                  else 
                     fprintf ( fp , "\n  constant = %lf", *wptr ) ;
                  } // For all output signals
               } // If trained
            else 
               fprintf ( fp , "Interrupted by user before weights found" ) ;
            fclose ( fp ) ;
            } // If audit log file opened OK
         } // If audit log on
      if (arma->trained)
         sprintf ( results , "Pooled RMS error = %lf", arma->error ) ;
      else 
         strcpy ( results , "Interrupted by user before weights found" ) ;
      return 0 ;
		}

   if (id == ID_PRED_ARMA_SHOCK) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " FOR " , 5 ))
            name[i] = rest[i] ;  // Parse shock signal name
         else
            break ;
         }
      j = i ;
      while (name[j-1] == ' ')  // Remove trailing blanks from shock name
         --j ;
      name[j] = 0 ;  // Shock signal name is here
      if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'FOR' also found
         rest += i+5 ;                      // Point to output signal name
         while (*rest == ' ')               // Remove leading blanks
            ++rest ;
         for (j=0 ; j<n_signals ; j++) {    // Look for that signal
            if (! strcmp ( signals[j]->name , rest )) {
               outsig = j ;
               break ;
               }
            }
         if (j == n_signals) {
            sprintf ( error , "Signal '%s' does not exist", rest ) ;
            return -1 ;
            }
         }
      else {
         strcpy ( error , "No output signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for shock signal
         if (! strcmp ( signals[i]->name , name )) {
            shocksig = i ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nARMA shock is %s for %s", name, rest ) ;
            fclose ( fp ) ;
            }
         }
      j = 0 ;   // Will flag if this output is found
		for (i=0 ; i<n_ins_outs ; i++) {
			if ((! ins_outs[i]->is_input)  &&  (ins_outs[i]->which == outsig)) {
            ins_outs[i]->shock = shocksig ;
            j = 1 ;   // Flag that output found
            }
         }
      if (! j) {
         sprintf ( error , "Signal '%s' is not an output in the model", rest ) ;
         return -1 ;
         }
      return 0 ;
      }

   if ((id == ID_PRED_ARMA_PREDICT)
    || (id == ID_PRED_ARMA_CONFIDENCE)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No arma name specified" ) ;
         return -1 ;
         }
      if (id == ID_PRED_ARMA_PREDICT) {
         if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
            strcpy ( error , "No created signal name specified" ) ;
            return -1 ;
            }
         }
      n = sscanf ( rest , "%d" , &ncases ) ;
      if ((! n)  ||  (ncases < 1)) {
         if (id == ID_PRED_ARMA_PREDICT)
            strcpy ( error , "Illegal predicted signal length specified" ) ;
         else 
            strcpy ( error , "Illegal future prediction distance specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip number of cases

		n_inputs = n_ma = n_outputs = 0 ; // Count inputs, MAs, and outputs
		for (i=0 ; i<n_ins_outs ; i++) {  // Check all ins and outs
			if (ins_outs[i]->is_input) {   // If this is an input
            if (! ins_outs[i]->minlag) {
               strcpy ( error , "Cannot have lag 0 for an ARMA input" ) ;
               return -1 ;
               }
				n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag + 1 ;
            }
			else {                         // This is an output
            if (ins_outs[i]->maxlag  &&  ! ins_outs[i]->minlag) {
               strcpy ( error , "Cannot have lag 0 MA term" ) ;
               return -1 ;
               }
				n_ma+= ins_outs[i]->maxlag - ins_outs[i]->minlag ;
            if (ins_outs[i]->minlag)
               ++n_ma ;  // Only lags are MA terms
            for (j=0 ; j<i ; j++) {     // Have we seen this output before?
               if ((! ins_outs[j]->is_input)  &&
                   (ins_outs[i]->which == ins_outs[j]->which))
                  break ;
               }
            if (i == j)       // If this is the first appearance of this output
               ++n_outputs ;  // Count it
            }
			}

		if (! (n_inputs + n_ma)) {
			strcpy ( error , "No inputs (or MAs) have been defined" ) ;
			return -1 ;
			}

		if (! n_outputs) {
			strcpy ( error , "No outputs have been defined" ) ;
         return -1 ;
			}

      for (i=0 ; i<n_ARMAs ; i++) {
         if (! strcmp ( ARMAs[i]->name , rest )) {
            arma = ARMAs[i] ;
            if (check_arma_compatibility ( n_inputs + n_ma , n_outputs ,
                                           arma_fixed , arma , error ))
               return -1 ;
            break ;
            }
         }
      if (i == n_ARMAs) {
         sprintf ( error , "ARMA '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (! arma->trained) {
         strcpy ( error , "This ARMA is not trained" ) ;
         return -1 ;
         }

      if (id == ID_PRED_ARMA_PREDICT) {
         if (n_outputs < misc_params.names->n) {
            sprintf( error , "ARMA has %d outputs, but %d names specified.",
                     n_outputs, misc_params.names->n ) ;
            return -1 ;
            }
         }

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_ARMA_PREDICT)
               fprintf ( fp , "\nPredict" ) ;
            else 
               fprintf ( fp , "\nConfidence" ) ;
            fprintf (fp,
               "\n with ARMA=%s having %d inputs, %d MA, and %d outputs" ,
               rest, n_inputs, n_ma, n_outputs ) ;
            for (i=0 ; i<n_ins_outs ; i++) {
               j = ins_outs[i]->which ;
               fprintf ( fp , "\n  %s", signals[j]->name ) ;
               if (ins_outs[i]->minlag  ||  ins_outs[i]->maxlag)
                  fprintf ( fp , "  Lag = %d", ins_outs[i]->minlag ) ;
               if (ins_outs[i]->maxlag > ins_outs[i]->minlag)
                  fprintf ( fp , " to %d", ins_outs[i]->maxlag ) ;
               if (ins_outs[i]->is_input)
                  fprintf ( fp , " (input)" ) ;
               else {
                  if (ins_outs[i]->minlag)
                     fprintf ( fp , " (MA output)" ) ;
                  else 
                     fprintf ( fp , " (output)" ) ;
                  }
               }
            if (id == ID_PRED_ARMA_PREDICT) {
               for (i=0 ; i<misc_params.names->n ; i++) {
                  if (misc_params.names->len[i])
                     fprintf ( fp , "\n  %s (predicted signal)",
                               misc_params.names->start[i] ) ;
                  else 
                     fprintf ( fp , "\n  {skip field}" ) ;
                  }
               }
            fclose ( fp ) ;
            }
         }

      if (id == ID_PRED_ARMA_PREDICT)
         ret = armapred ( ncases , arma , &misc_params ,
                          n_ins_outs , ins_outs ,
                          &n_signals , &signals ) ;
      else 
         ret = armaconf ( ncases , arma , n_ins_outs , ins_outs ,
                          signals , n_conf_comps , conf_comps ,
                          misc_params.conf_prob , &p , &t ) ;

      if (ret < 0) {
         strcpy ( error , "Insufficient memory to predict with ARMA" ) ;
         return -1 ;
         }
      else if (ret == 1) {
         strcpy ( error , "User interrupted" ) ;
         return -1 ;
         }
      else if (ret == 2) {
         strcpy ( error , "Too few cases for reliable results." ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_ARMA_PREDICT) {
               fprintf ( fp , "\n Pooled RMS error = %lf", sqrt(arma->error) ) ;
               if (ret > 0)
                  fprintf ( fp , " after user interruption" ) ;
               m = 0 ;    // Will count unique outputs
               for (iout=0 ; iout<n_ins_outs ; iout++) {  // Do all output sigs
                  if (ins_outs[iout]->is_input)
                     continue ;
                  for (j=0 ; j<iout ; j++) { // Have we seen this output before?
                     if ((! ins_outs[j]->is_input)  &&
                         (ins_outs[iout]->which == ins_outs[j]->which))
                        break ;
                     }
                  if (j < iout)      // If this output has been done already
                     continue ;      // Skip it
                  fprintf ( fp, "\n Output %s  (noise var=%lf  std=%lf)",
                     signals[ins_outs[iout]->which]->name,
                     arma->var[m], sqrt(arma->var[m]) ) ;
                  ++m ;
                  } // For all output signals
               } // If PREDICT
            else if (id == ID_PRED_ARMA_CONFIDENCE) {
               fprintf ( fp ,
            "\n  There is a 5%% chance each tail probability exceeds %.2lf %%",
                 p * 100.0 ) ;
               fprintf ( fp ,
            "\n  There is a %.2lf%% chance the interval exceeds %.2lf %%",
               t * 100.0 , 100.0 * (1.0 - 2.0 * (1.0 - misc_params.conf_prob)));
               }
            fclose ( fp ) ;
            } // If audit log file opened OK
         } // If audit log on
      if (id == ID_PRED_ARMA_PREDICT)
         sprintf ( results , "Pooled RMS error = %lf", arma->error ) ;
      return 0 ;
      }

   if (id == ID_PRED_ARMA_FIXED) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "Must be YES or NO" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "YES" ))
         arma_fixed = 1 ;
      else if (! strcmp ( rest , "NO" ))
         arma_fixed = 0 ;
      else {
         strcpy ( error , "Must be YES or NO" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nARMA fixed = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SAVE_ARMA) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No arma/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " TO " , 4 ))
            armaname[i] = rest[i] ;
         else
            break ;
         }
      armaname[i] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+4))
         strcpy ( filename , rest+i+4 ) ;
      else {
         strcpy ( error , "No 'TO file' name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_ARMAs ; i++) {
         if (! strcmp ( ARMAs[i]->name , armaname )) {
            if (! ARMAs[i]->trained) {
               strcpy ( error , "This ARMA is not trained" ) ;
               return -1 ;
               }
            ret = arma_save ( ARMAs[i] , filename ) ;
            switch (ret) {
               case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
               case 2: strcpy ( error , "Unable to write file" ) ; return -1 ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nSave ARMA = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "ARMA '%s' does not exist", armaname ) ;
      return -1 ;
      }

   if (id == ID_PRED_RESTORE_ARMA) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No ARMA/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " FROM " , 6 ))
            armaname[i] = rest[i] ;
         else
            break ;
         }
      armaname[i] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+6))
         strcpy ( filename , rest+i+6 ) ;
      else {
         strcpy ( error , "No 'FROM file' name specified" ) ;
         return -1 ;
         }
      arma = arma_restore ( armaname , filename , &ret ) ;
      if (arma == NULL) {
         switch (ret) {
            case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
            case 2: strcpy ( error , "Unable to read file" ) ; return -1 ;
            case 3: strcpy ( error , "Illegal file header" ) ; return -1 ;
            case 4: strcpy ( error , "Insufficient memory" ) ; return -1 ;
            }
         }
      for (i=0 ; i<n_ARMAs ; i++) {
         if (! strcmp ( ARMAs[i]->name , armaname )) {
            MEMTEXT ( "PROCESS: ARMA deleted for replacement" ) ;
            delete ARMAs[i] ;
            ARMAs[i] = arma ;
            break ;
            }
         }
      if (i == n_ARMAs) {
         MEMTEXT ( "PROCESS: ARMA array" ) ;
         if (n_ARMAs++)
            armas = (ARMA **) REALLOC ( ARMAs , n_ARMAs * sizeof(ARMA *)) ;
         else
            armas = (ARMA **) MALLOC ( n_ARMAs * sizeof(ARMA *)) ;
         if (armas == NULL) {
            strcpy ( error , "Insufficient memory" ) ;
            delete arma ;
            return -1 ;
            }
         ARMAs = armas ;
         ARMAs[n_ARMAs-1] = arma ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nRestore ARMA = %s  Pooled RMS error = %lf",
                      rest , sqrt(arma->error) ) ;
            m = 0 ;    // Will count unique outputs
            for (iout=0 ; iout<arma->nio ; iout++) {  // Do all output sigs
               if (arma->ins_outs[iout].is_input)
                  continue ;
               for (j=0 ; j<iout ; j++) { // Have we seen this output before?
                  if ((! arma->ins_outs[j].is_input)  &&
                      (arma->ins_outs[iout].which == arma->ins_outs[j].which))
                     break ;
                  }
               if (j < iout)      // If this output has been done already
                  continue ;      // Skip it
               wptr = arma->wts + m * arma->nvars ;
               fprintf ( fp, "\n Output %d  (noise var=%lf  std=%lf)",
                arma->ins_outs[iout].which , arma->var[m], sqrt(arma->var[m]) );
               ++m ;
               for (i=0 ; i<arma->nio ; i++) {  // Do all weights
                  if (arma->ins_outs[i].is_input) {
                     for (j=arma->ins_outs[i].minlag; j<=arma->ins_outs[i].maxlag; j++)
                        fprintf ( fp, "\n  %d (input)  Lag = %d  weight = %lf",
                                  arma->ins_outs[i].which, j, *wptr++ ) ;
                     } // If this is an input
                  else { // This is an output (MA if lagged)
                     for(j=arma->ins_outs[i].minlag; j<=arma->ins_outs[i].maxlag; j++){
                        if (j == 0)    // Only lags
                           continue ;  // are MA terms
                        fprintf ( fp , "\n  %d (MA)  Lag = %d  weight = %lf",
                                  arma->ins_outs[i].which, j, *wptr++ ) ;
                        }
                     } // Else this is an output (MA if lagged)
                  } // For all weights of this output signal
               if (arma->fixed)
                  fprintf ( fp , "\n  constant fixed at 0" ) ;
               else 
                  fprintf ( fp , "\n  constant = %lf", *wptr ) ;
               } // For all output signals
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if ((id == ID_PRED_CENTER)  ||  (id == ID_PRED_MEDIAN_CENTER)
    || (id == ID_PRED_DETREND)  ||  (id == ID_PRED_OFFSET)
    || (id == ID_PRED_SCALE)  ||  (id == ID_PRED_STANDARDIZE)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      if ((id == ID_PRED_OFFSET)  ||  (id == ID_PRED_SCALE)) {
         n = sscanf ( rest , "%lf" , &p ) ;
         if (! n) {
            strcpy ( error , "Illegal parameter specified" ) ;
            return -1 ;
            }
         while ((*rest == ' ')  ||  (*rest == '.')  ||  (*rest == '-')  ||
                ((*rest >= '0') && (*rest <= '9')))
            ++rest ;  // Skip parameter
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if ((id == ID_PRED_CENTER)  ||  (id == ID_PRED_MEDIAN_CENTER)) {
         if (sigptr->centered) {
            sprintf ( error , "Signal '%s' is already centered", rest ) ;
            return -1 ;
            }
         if (id == ID_PRED_CENTER)
            sigptr->center () ;
         else if (id == ID_PRED_MEDIAN_CENTER) {
            if (sigptr->median_center ()) {
               strcpy ( error , "Insufficient memory to median center" ) ;
               return -1 ;
               }
            }
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               if (id == ID_PRED_CENTER)
                  fprintf ( fp , "\nCenter %s", rest ) ;
               else if (id == ID_PRED_MEDIAN_CENTER)
                  fprintf ( fp , "\nMedian center %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_DETREND) {
         if (sigptr->detrended) {
            sprintf ( error , "Signal '%s' is already detrended", rest ) ;
            return -1 ;
            }
         sigptr->detrend () ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nDetrend %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_OFFSET) {
         if (sigptr->offsetted) {
            sprintf ( error , "Signal '%s' is already offsetted", rest ) ;
            return -1 ;
            }
         sigptr->offset ( p ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nOffset %s by %lf", rest, p ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_SCALE) {
         if (sigptr->scaled) {
            sprintf ( error , "Signal '%s' is already scaled", rest ) ;
            return -1 ;
            }
         sigptr->scale ( p ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nScale %s by %lf", rest, p ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_STANDARDIZE) {
         if (sigptr->standardized) {
            sprintf ( error , "Signal '%s' is already standardized", rest ) ;
            return -1 ;
            }
         sigptr->standardize () ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nStandardize %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      return 0 ;
      }

   if ((id == ID_PRED_UNDO_CENTER)  ||  (id == ID_PRED_UNDO_DETREND)
    || (id == ID_PRED_UNDO_OFFSET)  ||  (id == ID_PRED_UNDO_SCALE)
    ||  (id == ID_PRED_UNDO_STANDARDIZE)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " PER " , 5 ))
            name[i] = rest[i] ;  // Parse source signal name
         else
            break ;
         }
      j = i ;
      while (name[j-1] == ' ')  // Remove trailing blanks from name
         --j ;
      name[j] = 0 ;  // Source signal name is here
      if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'PER' also found
         rest += i+5 ;                      // Point to PER signal name
         while (*rest == ' ')               // Remove leading blanks
            ++rest ;
         for (j=0 ; j<n_signals ; j++) {    // Look for that signal
            if (! strcmp ( signals[j]->name , rest )) {
               sigptr2 = signals[j] ;  // Point to PER signal
               break ;
               }
            }
         if (j == n_signals) {
            sprintf ( error , "Signal '%s' does not exist", rest ) ;
            return -1 ;
            }
         }
      else
         sigptr2 = NULL ;   // Flag that there is no PER signal
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , name )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }
      if (id == ID_PRED_UNDO_CENTER) {
         sigptr->undo_center ( sigptr2 ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nUndo center of %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_UNDO_DETREND) {
         sigptr->undo_detrend ( sigptr2 ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nUndo detrend of %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_UNDO_OFFSET) {
         sigptr->undo_offset ( sigptr2 ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nUndo offset of %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_UNDO_SCALE) {
         sigptr->undo_scale ( sigptr2 ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nUndo scale of %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      else if (id == ID_PRED_UNDO_STANDARDIZE) {
         sigptr->undo_standardize ( sigptr2 ) ;
         if (strlen ( audit_log )) {
            if ((fp = fopen ( audit_log , "at" )) != NULL) {
               fprintf ( fp , "\nUndo standardize of %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               fclose ( fp ) ;
               }
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_DIFFERENCE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No degree or signal name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &degree ) ;
      if ((! n)  ||  (degree < 1)  ||  (degree > 3)) {
         strcpy ( error , "Illegal differencing degree specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip degree
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < degree+1) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      if (sigptr->differenced) {
         sprintf ( error , "Signal '%s' is already differenced", rest ) ;
         return -1 ;
         }
      sigptr->difference ( degree ) ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nDifference %s with degree=%d", rest, degree ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_UNDO_DIFFERENCE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " PER " , 5 ))
            name[i] = rest[i] ;  // Parse source signal name
         else
            break ;
         }
      j = i ;
      while (name[j-1] == ' ')  // Remove trailing blanks from name
         --j ;
      name[j] = 0 ;  // Source signal name is here
      if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'PER' also found
         rest += i+5 ;                      // Point to PER signal name
         while (*rest == ' ')               // Remove leading blanks
            ++rest ;
         for (j=0 ; j<n_signals ; j++) {    // Look for that signal
            if (! strcmp ( signals[j]->name , rest )) {
               sigptr2 = signals[j] ;  // Point to PER signal
               break ;
               }
            }
         if (j == n_signals) {
            sprintf ( error , "Signal '%s' does not exist", rest ) ;
            return -1 ;
            }
         }
      else
         sigptr2 = NULL ;   // Flag that there is no PER signal
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , name )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }
      if (sigptr->undo_difference ( sigptr2 )) {
         sprintf ( error , "Insufficient memory for undo difference" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nUndo difference of %s", name ) ;
            if (sigptr2 != NULL)
               fprintf ( fp , " PER %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SEASONAL_DIFFERENCE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No period or signal name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &period ) ;
      if ((! n)  ||  (period < 1)) {
         strcpy ( error , "Illegal seasonal period specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip period
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < 2*period) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      if (sigptr->seasonal) {
         sprintf ( error , "Signal '%s' is already seasonal differenced", rest);
         return -1 ;
         }
      if (sigptr->seasonal_diff ( period )) {
         sprintf ( error , "Insufficient memory for seasonal difference" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nSeasonal difference %s with period=%d",
                      rest, period ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_UNDO_SEASONAL_DIFFERENCE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " PER " , 5 ))
            name[i] = rest[i] ;  // Parse source signal name
         else
            break ;
         }
      j = i ;
      while (name[j-1] == ' ')  // Remove trailing blanks from name
         --j ;
      name[j] = 0 ;  // Source signal name is here
      if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'PER' also found
         rest += i+5 ;                      // Point to PER signal name
         while (*rest == ' ')               // Remove leading blanks
            ++rest ;
         for (j=0 ; j<n_signals ; j++) {    // Look for that signal
            if (! strcmp ( signals[j]->name , rest )) {
               sigptr2 = signals[j] ;  // Point to PER signal
               break ;
               }
            }
         if (j == n_signals) {
            sprintf ( error , "Signal '%s' does not exist", rest ) ;
            return -1 ;
            }
         }
      else
         sigptr2 = NULL ;   // Flag that there is no PER signal
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , name )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }
      if (sigptr->undo_seasonal_diff ( sigptr2 )) {
         sprintf ( error , "Insufficient memory for undo difference" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nUndo seasonal difference of %s", name ) ;
            if (sigptr2 != NULL)
               fprintf ( fp , " PER %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if ((id == ID_PRED_LOG)  ||  (id == ID_PRED_EXP)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (id == ID_PRED_LOG)
         sigptr->siglog () ;
      else if (id == ID_PRED_EXP)
         sigptr->sigexp () ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_LOG)
               fprintf ( fp , "\nLog %s", rest ) ;
            else if (id == ID_PRED_EXP)
               fprintf ( fp , "\nExp %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if ((id == ID_PRED_CONFIDENCE_CENTER)
    || (id == ID_PRED_CONFIDENCE_DETREND)
    || (id == ID_PRED_CONFIDENCE_OFFSET)
    || (id == ID_PRED_CONFIDENCE_SCALE)
    || (id == ID_PRED_CONFIDENCE_STANDARDIZE)
    || (id == ID_PRED_CONFIDENCE_DIFFERENCE)
    || (id == ID_PRED_CONFIDENCE_SEASONAL)
    || (id == ID_PRED_CONFIDENCE_LOG)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      if ((id == ID_PRED_CONFIDENCE_CENTER)
       || (id == ID_PRED_CONFIDENCE_DETREND)
       || (id == ID_PRED_CONFIDENCE_OFFSET)
       || (id == ID_PRED_CONFIDENCE_SCALE)
       || (id == ID_PRED_CONFIDENCE_STANDARDIZE)
       || (id == ID_PRED_CONFIDENCE_DIFFERENCE)
       || (id == ID_PRED_CONFIDENCE_SEASONAL)) {
         for (i=0 ; i<strlen(rest) ; i++) {
            if (strncmp ( rest+i , " PER " , 5 ))
               name[i] = rest[i] ;  // Parse source signal name
            else
               break ;
            }
         j = i ;
         while (name[j-1] == ' ')  // Remove trailing blanks from name
            --j ;
         name[j] = 0 ;  // Source signal name is here
         if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'PER' also found
            rest += i+5 ;                      // Point to PER signal name
            while (*rest == ' ')               // Remove leading blanks
               ++rest ;
            for (j=0 ; j<n_signals ; j++) {    // Look for that signal
               if (! strcmp ( signals[j]->name , rest )) {
                  sigptr2 = signals[j] ;  // Point to PER signal
                  break ;
                  }
               }
            if (j == n_signals) {
               sprintf ( error , "Signal '%s' does not exist", rest ) ;
               return -1 ;
               }
            }
         else
            sigptr2 = NULL ;   // Flag that there is no PER signal
         }
      else
         strcpy ( name , rest ) ;      // No PER signal

      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , name ))
            break ;
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }

      MEMTEXT ( "PROCESS: conf_comps" ) ;
      if (n_conf_comps)
         ccptr = (ConfComp *) REALLOC ( conf_comps ,
                                        (n_conf_comps+1) * sizeof(ConfComp)) ;
      else 
         ccptr = (ConfComp *) MALLOC ( sizeof(ConfComp) ) ;

      if (ccptr == NULL) {
         sprintf ( error , "Insufficient memory for CONFIDENCE compensation");
         return -1 ;
         }

      conf_comps = ccptr ;
      ccptr += n_conf_comps++ ;
      ccptr->signum = i ;
      if (sigptr2 != NULL)
         ccptr->persig = sigptr2 ;
      else 
         ccptr->persig = signals[i] ;
// It would be nice to verify that these ops were done previously!
      if (id == ID_PRED_CONFIDENCE_CENTER)
         ccptr->which = CCcenter ;
      else if (id == ID_PRED_CONFIDENCE_DETREND)
         ccptr->which = CCdetrend ;
      else if (id == ID_PRED_CONFIDENCE_OFFSET)
         ccptr->which = CCoffset ;
      else if (id == ID_PRED_CONFIDENCE_SCALE)
         ccptr->which = CCscale ;
      else if (id == ID_PRED_CONFIDENCE_STANDARDIZE)
         ccptr->which = CCstandardize ;
      else if (id == ID_PRED_CONFIDENCE_DIFFERENCE)
         ccptr->which = CCdifference ;
      else if (id == ID_PRED_CONFIDENCE_SEASONAL)
         ccptr->which = CCseasonal ;
      else if (id == ID_PRED_CONFIDENCE_LOG)
         ccptr->which = CClog ;

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_CONFIDENCE_CENTER) {
               fprintf ( fp , "\nConfidence Center %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            else if (id == ID_PRED_CONFIDENCE_DETREND) {
               fprintf ( fp , "\nConfidence Detrend %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            else if (id == ID_PRED_CONFIDENCE_OFFSET) {
               fprintf ( fp , "\nConfidence Offset %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            else if (id == ID_PRED_CONFIDENCE_SCALE) {
               fprintf ( fp , "\nConfidence Scale %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            else if (id == ID_PRED_CONFIDENCE_STANDARDIZE) {
               fprintf ( fp , "\nConfidence Standardize %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            else if (id == ID_PRED_CONFIDENCE_DIFFERENCE) {
               fprintf ( fp , "\nConfidence Difference %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            else if (id == ID_PRED_CONFIDENCE_SEASONAL) {
               fprintf ( fp , "\nConfidence Seasonal difference %s", name ) ;
               if (sigptr2 != NULL)
                  fprintf ( fp , " PER %s", rest ) ;
               }
            if (id == ID_PRED_CONFIDENCE_LOG)
               fprintf ( fp , "\nConfidence Log %s", name ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_CONFIDENCE) {
      if (conf_comps != NULL) {
         MEMTEXT ( "PROCESS: conf_comps" ) ;
         FREE ( conf_comps ) ;
         conf_comps = NULL ;
         n_conf_comps = 0 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nClear Confidence Compensation" ) ;
            fclose ( fp ) ;
            }
         }
      }

   if (id == ID_PRED_CONFIDENCE_PROB) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No probability specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 50.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal CONFIDENCE PROBABILITY = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.conf_prob = 0.01 * p ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nConfidence probability = %.3lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if ((id == ID_PRED_ADD)  ||  (id == ID_PRED_SUBTRACT)
    || (id == ID_PRED_MULTIPLY)  ||  (id == ID_PRED_DIVIDE)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " AND " , 5 ))
            name[i] = rest[i] ;  // Parse first signal name
         else
            break ;
         }
      j = i ;
      while (name[j-1] == ' ')  // Remove trailing blanks from first name
         --j ;
      name[j] = 0 ;  // first signal name is here
      if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'AND' also found
         rest += i+5 ;                      // Point to second signal name
         while (*rest == ' ')               // Remove leading blanks
            ++rest ;
         for (j=0 ; j<n_signals ; j++) {    // Look for that signal
            if (! strcmp ( signals[j]->name , rest )) {
               sigptr2 = signals[j] ;
               break ;
               }
            }
         if (j == n_signals) {
            sprintf ( error , "Signal '%s' does not exist", rest ) ;
            return -1 ;
            }
         }
      else {
         strcpy ( error , "No second signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for first signal
         if (! strcmp ( signals[i]->name , name )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_ADD)
               fprintf ( fp , "\nAdd %s and %s to create:", name, rest ) ;
            else if (id == ID_PRED_SUBTRACT)
               fprintf ( fp , "\nSubtract %s and %s to create:", name, rest ) ;
            else if (id == ID_PRED_MULTIPLY)
               fprintf ( fp , "\nMultiply %s and %s to create:", name, rest ) ;
            else if (id == ID_PRED_DIVIDE)
               fprintf ( fp , "\nDivide %s and %s to create:", name, rest ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (sigptr->n != sigptr2->n) {
         sprintf ( error , "Signal lengths not equal (%d, %d)",
                   sigptr->n, sigptr2->n ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (combine ( &misc_params , id , sigptr , sigptr2 ,
                    &n_signals , &signals , error ))
         return -1 ;
      return 0 ;
      }

   if (id == ID_PRED_ORTHOG_TYPE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No orthogonalization type specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "PRINCIPAL COMPONENTS" ))
         misc_params.orthog_type = 1 ;
      else if (! strcmp ( rest , "CENTROIDS" ))
         misc_params.orthog_type = 2 ;
      else if (! strcmp ( rest , "DISCRIMINANT" ))
         misc_params.orthog_type = 3 ;
      else {
         sprintf ( error , "Illegal ORTHOGONALIZATION TYPE = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nORTHOGONALIZATION TYPE = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ORTHOG_FACTORS) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No number of factors specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &m ) ;
      if ((m < 0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ORTHOGONALIZATION FACTORS = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.orthog_nfac = m ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nORTHOGONALIZATION FACTORS = %d", m ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ORTHOG_LIMIT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No limit specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%lf" , &p ) ;
      if ((p <= 0.0)  ||  (p > 100.0)  ||  (n <= 0)) {
         sprintf ( error , "Illegal ORTHOGONALIZATION LIMIT = %s", rest ) ;
         return -1 ;
         }
      else
         misc_params.orthog_lim = p / 100.0 ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nORTHOGONALIZATION LIMIT = %lf", p ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_ORTHOG_STD) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No orthogonalization standardization specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "YES" ))
         misc_params.orthog_std = 1 ;
      else if (! strcmp ( rest , "NO" ))
         misc_params.orthog_std = 0 ;
      else {
         sprintf ( error , "Illegal ORTHOGONALIZATION STANDARDIZE = %s", rest);
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nORTHOGONALIZATION STANDARDIZE = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_DEFINE_ORTHOG) {
      if ((! rest)  ||  ((n=strlen (rest)) == 0)) {
         strcpy ( error , "No orthogonalization name specified" ) ;
         return -1 ;
         }
      if ((trnset == NULL)  ||  (trnset->ntrain == 0)) {
         strcpy ( error ,
                 "Cannot DEFINE ORTHOGONALIZATION; No training set exists." ) ;
         return -1 ;
         }
      if (trnset->output_mode != OUTMOD_CLASSIFICATION) {
         strcpy ( error , "Training set must be in CLASSIFICATION mode" ) ;
         return -1 ;
         }
      if (! trnset->n_outputs) {
         strcpy ( error ,
            "Cannot DEFINE ORTHOGONALIZATION; No outputs (or all REJECT)." ) ;
         return -1 ;
         }
      if (trnset->n_outputs  !=  n_classes) {
         sprintf ( error , "%d class names exist, but training set has %d.",
                   n_classes, trnset->n_outputs ) ;
         return -1 ;
         }
      if ((misc_params.orthog_type > 1)  &&  (trnset->n_outputs < 2)) {
         sprintf ( error , "Must have more than one class for this type." ) ;
         return -1 ;
         }
      for (i=0 ; i<trnset->n_outputs ; i++) {
         if (trnset->nper[i] == 0) {
            sprintf ( error , "Class %s has no cases.", classes[i]+1 ) ;
            return -1 ;
            }
         }
      for (i=0 ; i<trnset->ntrain ; i++) {
         dptr = trnset->data + trnset->size * i ;
         if ((int) dptr[trnset->n_inputs]  ==  0) {
            strcpy(error, "DEFINE ORTHOGONALIZATION cannot have REJECT class.");
            return -1 ;
            }
         }

      MEMTEXT ( "PROCESS: new Orthog" ) ;
      orthog = new Orthog ( rest , trnset , &misc_params ) ;
      if ((orthog == NULL)  ||  (orthog->ok == 0)) {
         if (orthog != NULL)
            delete orthog ;
         strcpy ( error , "Insufficient memory to DEFINE ORTHOGONALIZATION." ) ;
         return -1 ;
         }

      for (i=0 ; i<n_orthogs ; i++) {    // Is this name here?
         if (! strcmp ( orthogs[i]->name , rest ))
            break ;
         }
      if (i < n_orthogs) {     // This name is already here
         delete orthogs[i] ;   // Delete the old one
         orthogs[i] = orthog ; // And put in the new
         }
      else {
         MEMTEXT ( "PROCESS: orthogs array" ) ;
         if (orthogs)
            oth = (Orthog **) REALLOC ( orthogs ,
                                       (n_orthogs+1) * sizeof(Orthog *) ) ;
         else 
            oth = (Orthog **) MALLOC ( (n_orthogs+1) * sizeof(Orthog *) ) ;
         if (oth == NULL) {
            strcpy ( error , "Insufficient memory to append orthog." ) ;
            delete orthog ;
            return -1 ;
            }
         orthogs = oth ;
         orthogs[n_orthogs++] = orthog ;
         }

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nDefine orthogonalization = %s", rest ) ;
            if (orthog->type == 3)
               orthog->discrim->eigen ( &dptr , &wptr ) ;
            else 
               orthog->princo->eigen ( &dptr , &wptr ) ;
            n = orthog->nin ;
            sum = 0.0 ;
            for (i=0 ; i<n ; i++)
               sum += dptr[i] ;
            fprintf ( fp, "\nPercent   ----->" ) ;
            for (i=0 ; i<orthog->nfacs ; i++)
               fprintf ( fp, " %9.4lf", dptr[i] / sum * 100.0 ) ;
            for (i=0 ; i<orthog->nin ; i++) {
               fprintf ( fp , "\nVariable %2d (%2d)", i+1, orthog->lags[i] ) ;
               for (m=0 ; m<orthog->nfacs ; m++)
                  fprintf ( fp , " %9.4lf", wptr[i*n+m] ) ;
               }
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_APPLY_ORTHOG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No orthogonalization name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_orthogs ; i++) {
         if (! strcmp ( orthogs[i]->name , rest )) {
            orthog = orthogs[i] ;
            break ;
            }
         }
      if (i == n_orthogs) {
         sprintf ( error , "Orthogonalization '%s' does not exist", rest ) ;
         return -1 ;
         }

		n_inputs = 0 ;
		for (i=0 ; i<n_ins_outs ; i++) {
			if (ins_outs[i]->is_input)
				n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
			}

		if (! n_inputs) {
			strcpy ( error , "No inputs have been defined" ) ;
			return -1 ;
			}

      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }

      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nAPPLY ORTHOGONALIZATION = %s", rest ) ;
            for (i=0 ; i<n_ins_outs ; i++) {
               if (ins_outs[i]->is_input) {
                  j = ins_outs[i]->which ;
                  fprintf ( fp , "\n  %s", signals[j]->name ) ;
                  if (ins_outs[i]->minlag  ||  ins_outs[i]->maxlag)
                     fprintf ( fp , "  Lag = %d", ins_outs[i]->minlag ) ;
                  if (ins_outs[i]->maxlag > ins_outs[i]->minlag)
                     fprintf ( fp , " to %d", ins_outs[i]->maxlag ) ;
                  fprintf ( fp , " (input)" ) ;
                  }
               }
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s (created signal)",
                            misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }

      if (orthog->nin != n_inputs) {
         sprintf( error , "Signal inputs (%d) 'not equal to orthog inputs (%d)",
                  n_inputs, orthog->nin ) ;
         return -1 ;
         }

      ret = orthog->transform ( n_ins_outs , ins_outs , &n_signals ,
                                &signals , &misc_params ) ;

      if (ret < 0) {
         strcpy ( error , "Insufficient memory to APPLY ORTHOGNALIZATION." ) ;
         return -1 ;
         }
      return 0 ;
      }

   if (id == ID_PRED_SAVE_ORTHOG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No orthog/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " TO " , 4 ))
            orthname[i] = rest[i] ;
         else
            break ;
         }
      orthname[i] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+4))
         strcpy ( filename , rest+i+4 ) ;
      else {
         strcpy ( error , "No 'TO file' name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_orthogs ; i++) {
         if (! strcmp ( orthogs[i]->name , orthname )) {
            ret = orth_save ( orthogs[i] , filename ) ;
            switch (ret) {
               case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
               case 2: strcpy ( error , "Unable to write file" ) ; return -1 ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nSave ORTHOGONALIZATION = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "ORTHOGONALIZATION '%s' does not exist", orthname ) ;
      return -1 ;
      }

   if (id == ID_PRED_RESTORE_ORTHOG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No orthog/file name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<strlen(rest) ; i++) {
         if (strncmp ( rest+i , " FROM " , 6 ))
            orthname[i] = rest[i] ;
         else
            break ;
         }
      orthname[i] = 0 ;
      if ((i < strlen (rest))  &&  strlen ( rest+i+6))
         strcpy ( filename , rest+i+6 ) ;
      else {
         strcpy ( error , "No 'FROM file' name specified" ) ;
         return -1 ;
         }
      orthog = orth_restore ( orthname , filename , &ret ) ;
      if (arma == NULL) {
         switch (ret) {
            case 1: strcpy ( error , "Unable to open file" ) ; return -1 ;
            case 2: strcpy ( error , "Unable to read file" ) ; return -1 ;
            case 3: strcpy ( error , "Illegal file header" ) ; return -1 ;
            case 4: strcpy ( error , "Insufficient memory" ) ; return -1 ;
            }
         }
      for (i=0 ; i<n_orthogs ; i++) {
         if (! strcmp ( orthogs[i]->name , orthname )) {
            MEMTEXT ( "PROCESS: Orthog deleted for replacement" ) ;
            delete orthogs[i] ;
            orthogs[i] = orthog ;
            break ;
            }
         }
      if (i == n_orthogs) {
         MEMTEXT ( "PROCESS: Orthog array" ) ;
         if (n_orthogs++)
            oth = (Orthog **) REALLOC ( orthogs , n_orthogs * sizeof(Orthog *));
         else
            oth = (Orthog **) MALLOC ( n_orthogs * sizeof(Orthog *)) ;
         if (oth == NULL) {
            strcpy ( error , "Insufficient memory" ) ;
            delete orthog ;
            return -1 ;
            }
         orthogs = oth ;
         orthogs[n_orthogs-1] = orthog ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nRestore ORTHOGONALIZATION = %s", rest ) ;
            if (orthog->type == 3)
               orthog->discrim->eigen ( &dptr , &wptr ) ;
            else 
               orthog->princo->eigen ( &dptr , &wptr ) ;
            n = orthog->nin ;
            sum = 0.0 ;
            for (i=0 ; i<n ; i++)
               sum += dptr[i] ;
            fprintf ( fp, "\nPercent   ----->" ) ;
            for (i=0 ; i<orthog->nfacs ; i++)
               fprintf ( fp, " %9.4lf", dptr[i] / sum * 100.0 ) ;
            for (i=0 ; i<orthog->nin ; i++) {
               fprintf ( fp , "\nVariable %2d (%2d)", i+1, orthog->lags[i] ) ;
               for (m=0 ; m<orthog->nfacs ; m++)
                  fprintf ( fp , " %9.4lf", wptr[i*n+m] ) ;
               }
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_ORTHOGS) {
      if (n_orthogs) {
         MEMTEXT ( "PROCESS: orthogs" ) ;
         for (i=0 ; i<n_orthogs ; i++) {
            if (orthogs[i] != NULL)
               delete orthogs[i] ;
            }
         FREE ( orthogs ) ;
         orthogs = NULL ;
         n_orthogs = 0 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nClear Orthogonalizations" ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_CLEAR_ORTHOG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No orthogonalization name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_orthogs ; i++) {
         if (! strcmp ( orthogs[i]->name , rest )) {
            MEMTEXT ( "PROCESS: orthog" ) ;
            delete orthogs[i] ;
            for (j=i+1 ; j<n_orthogs ; j++)
               orthogs[j-1] = orthogs[j] ;
            --n_orthogs ;
            if (n_orthogs)
               orthogs = (Orthog **) REALLOC ( orthogs ,
                                               n_orthogs * sizeof(Orthog *)) ;
            else {
               MEMTEXT ( "PROCESS: orthog array" ) ;
               FREE ( orthogs ) ;
               orthogs = NULL ;
               }
            if (strlen ( audit_log )) {
               if ((fp = fopen ( audit_log , "at" )) != NULL) {
                  fprintf ( fp , "\nClear Orthogonalization = %s", rest ) ;
                  fclose ( fp ) ;
                  }
               }
            return 0 ;
            }
         }
      sprintf ( error , "ORTHOGONALIZATION '%s' does not exist", rest ) ;
      return -1 ;
      }

   if ((id == ID_PRED_AUTOCORR)  ||  (id == ID_PRED_CROSSCORR)
    || (id == ID_PRED_P_AUTOCORR)  ||  (id == ID_PRED_P_CROSSCORR)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No count or signal name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &ncases ) ;
      if ((! n)  ||  (ncases < 1)) {
         strcpy ( error , "Illegal computed signal length specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip number of cases
      if ((id == ID_PRED_AUTOCORR)  ||  (id == ID_PRED_P_AUTOCORR))
         strcpy ( name , rest ) ;
      else if ((id == ID_PRED_CROSSCORR)  ||  (id == ID_PRED_P_CROSSCORR)) {
         for (i=0 ; i<strlen(rest) ; i++) {
            if (strncmp ( rest+i , " AND " , 5 ))
               name[i] = rest[i] ;  // Parse first signal name
            else
               break ;
            }
         j = i ;
         while (name[j-1] == ' ')  // Remove trailing blanks from first name
            --j ;
         name[j] = 0 ;  // first signal name is here
         if ((i < strlen (rest))  &&  strlen ( rest+i+5)) { // If 'AND' found
            rest += i+5 ;                      // Point to second signal name
            while (*rest == ' ')               // Remove leading blanks
               ++rest ;
            for (j=0 ; j<n_signals ; j++) {    // Look for that signal
               if (! strcmp ( signals[j]->name , rest )) {
                  sigptr2 = signals[j] ;
                  break ;
                  }
               }
            if (j == n_signals) {
               sprintf ( error , "Signal '%s' does not exist", rest ) ;
               return -1 ;
               }
            }
         else {
            strcpy ( error , "No second signal name specified" ) ;
            return -1 ;
            }
         } // Else CROSSCORR
      for (i=0 ; i<n_signals ; i++) {  // Look for first signal
         if (! strcmp ( signals[i]->name , name )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", name ) ;
         return -1 ;
         }
      if ((ncases >= sigptr->n)  ||
          ((id == ID_PRED_CROSSCORR)  &&  (ncases >= sigptr2->n))) {
         sprintf ( error , "Signal too short" ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_AUTOCORR)
               fprintf ( fp , "\nAUTOCORRELATION %s to create:", name ) ;
            else if (id == ID_PRED_P_AUTOCORR)
               fprintf ( fp , "\nPARTIAL AUTOCORRELATION %s to create:",
                         name ) ;
            else if (id == ID_PRED_CROSSCORR)
               fprintf ( fp , "\nCROSSCORRELATION %s and %s to create:",
                              name, rest ) ;
            else if (id == ID_PRED_P_CROSSCORR)
               fprintf ( fp , "\nPARTIAL CROSSCORRELATION %s and %s to create:",
                              name, rest ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (autocorr ( &misc_params , id , ncases , sigptr , sigptr2 ,
                     &n_signals , &signals , error , &dptr ))
         return -1 ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\n  " ) ;
            for (i=0 ; i<ncases ; i++) {
               if (i > 12)
                  break ;
               fprintf ( fp , " %.3lf", dptr[i] ) ;
               }
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_COPY) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &ncases ) ;
      if (! n)        // If no count specified
         ncases = 0 ; // Copy full signal
      else if (! ncases) {
         strcpy ( error , "Illegal count" ) ;
         return -1 ;
         }
      while ((*rest == '-')  ||  (*rest == ' ')  ||
             ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip number of cases
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nCOPY %s (len=%d) to create:", rest, ncases ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (copy ( &misc_params , ncases , sigptr , &n_signals , &signals, error))
         return -1 ;
      return 0 ;
      }

   if (id == ID_PRED_INTEGRATE) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No period or signal name specified" ) ;
         return -1 ;
         }
      n = sscanf ( rest , "%d" , &period ) ;
      if (! n)
         period = 1 ;
      else if (period < 1) {
         strcpy ( error , "Illegal period specified" ) ;
         return -1 ;
         }
      while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip period
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;  // Point to source signal
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < period+1) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      sigptr->integrate ( period ) ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nIntegrate %s with period=%d", rest, period ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SPECTRUM) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (spectrum ( &misc_params , sigptr , &n_signals , &signals ,
                     &t , &p , error ))
         return -1 ;
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nSPECTRUM of %s to create:", rest ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (i >= 5)
                  break ;
               switch (i) {
                  case 0:
                     fprintf ( fp , "\n  REAL : " ) ;  break ;
                  case 1:
                     fprintf ( fp , "\n  IMAG : " ) ;  break ;
                  case 2:
                     fprintf ( fp , "\n  POWER: " ) ;  break ;
                  case 3:
                     fprintf ( fp , "\n  PHASE: " ) ;  break ;
                  case 4:
                     fprintf ( fp , "\n  DEV: " ) ;  break ;
                  }
               if (misc_params.names->len[i]) {
                  fprintf ( fp , "%s", misc_params.names->start[i] ) ;
                  if (i == 4)
                     fprintf ( fp , "  DMAX = %.4lf  (alpha = %.3lf)", t, p ) ;
                  }
               else 
                  fprintf ( fp , "{skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SPECTRUM_WINDOW) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No spectrum window specified" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "NONE" ))
         misc_params.spectrum_window = 0 ;
      else if (! strcmp ( rest , "WELCH" ))
         misc_params.spectrum_window = 1 ;
      else {
         sprintf ( error , "Illegal SPECTRUM WINDOW = %s", rest ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nSpectrum window = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_SAVGOL) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      while (*rest == ' ')
         ++rest ;
      if ((*rest < '0') || (*rest > '9'))
         ncases = 0 ;  // Flag to use defaults
      else {
         n = sscanf ( rest , "%d" , &ncases ) ;
         if (!n  ||  (ncases < 1)) {  // If no count specified
            strcpy ( error , "Illegal first parameter" ) ;
            return -1 ;
            }
         while ((*rest >= '0') && (*rest <= '9'))
            ++rest ;  // Skip half length
         while (*rest == ' ')
            ++rest ;  // Skip following blanks
         n = sscanf ( rest , "%d" , &degree ) ;
         if (!n  ||  (degree <= 1)) {  // If no degree specified
            strcpy ( error , "Illegal second parameter" ) ;
            return -1 ;
            }
         while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
            ++rest ;  // Skip degree
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < 10) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (ncases == 0) {
         ncases = sigptr->n / 40 ;
         if (ncases < 2)
            ncases = 2 ;
         degree = 4 ;
         if (ncases < 5)
            degree = 2 ;
         }
      if (degree > ncases) {
         strcpy ( error , "Degree to large for half-length" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp ,
               "\nSavitzky-Golay filter %s (halflen=%d  degree=%d) to create:",
                rest, ncases, degree ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (savgol ( &misc_params , ncases , degree , sigptr ,
                   &n_signals , &signals, error))
         return -1 ;
      return 0 ;
      }

   if (id == ID_PRED_MAXENT) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      while (*rest == ' ')
         ++rest ;
      if ((*rest < '0') || (*rest > '9'))
         ncases = 0 ;  // Flag to use defaults
      else {
         n = sscanf ( rest , "%d" , &ncases ) ;
         if (!n  ||  (ncases <= 1)) {  // If no count specified
            strcpy ( error , "Illegal first parameter" ) ;
            return -1 ;
            }
         while ((*rest >= '0') && (*rest <= '9'))
            ++rest ;  // Skip number of cases
         while (*rest == ' ')
            ++rest ;  // Skip following blanks
         n = sscanf ( rest , "%d" , &degree ) ;
         if (!n  ||  (degree <= 1)) {  // If no degree specified
            strcpy ( error , "Illegal second parameter" ) ;
            return -1 ;
            }
         while ((*rest == ' ')  ||  ((*rest >= '0') && (*rest <= '9')))
            ++rest ;  // Skip degree
         }
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (ncases == 0) {
         ncases = sigptr->n * 4 ;
         if (ncases < 100)
            ncases = 100 ;
         degree = sigptr->n / 50 ;
         if (degree > 100)
            degree = 100 ;
         if (degree < 10)
            degree = 10 ;
         }
      if (degree > (sigptr->n/2-5)) {  // Arbitrary but reasonable
         strcpy ( error , "Order too large" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp ,
               "\nMaximum entropy spectrum %s (len=%d  order=%d) to create:",
                rest, ncases, degree ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (maxent ( &misc_params , ncases , degree , sigptr ,
                   &n_signals , &signals, error))
         return -1 ;
      return 0 ;
      }

   if ((id == ID_PRED_LOWPASS)  ||  (id == ID_PRED_HIGHPASS)  ||
       (id == ID_PRED_BANDPASS)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      while (*rest == ' ')
         ++rest ;
      n = sscanf ( rest , "%lf" , &freq ) ;
      if (!n || (freq < 0.0) || (freq > 0.5)) {  // If no frequency specified
         strcpy ( error , "Illegal frequency" ) ;
         return -1 ;
         }
      while ((*rest == '.')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip frequency
      while (*rest == ' ')
         ++rest ;  // Skip following blanks
      n = sscanf ( rest , "%lf" , &width ) ;
      if (!n || (width <= 0.0) || (width >= 0.5)) {  // If no width specified
         strcpy ( error , "Illegal width" ) ;
         return -1 ;
         }
      while ((*rest == ' ') || (*rest == '.') || ((*rest>='0') && (*rest<='9')))
         ++rest ;  // Skip width
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < 2) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_LOWPASS)
               fprintf ( fp , "\nLowpass" ) ;
            else if (id == ID_PRED_HIGHPASS)
               fprintf ( fp , "\nHighpass" ) ;
            else if (id == ID_PRED_BANDPASS)
               fprintf ( fp , "\nBandpass" ) ;
            fprintf ( fp , " filter %s (freq=%.4lf  width=%.4lf) to create:",
                      rest, freq, width ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (filt_sig ( &misc_params , id , sigptr , freq , width ,
                     &n_signals , &signals, error ))
         return -1 ;
      return 0 ;
      }

   if ((id == ID_PRED_QMF)  ||  (id == ID_PRED_MORLET)) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      while (*rest == ' ')
         ++rest ;
      n = sscanf ( rest , "%lf" , &freq ) ;
      if (!n || (freq < 0.0) || (freq > 0.5)) {  // If no frequency specified
         strcpy ( error , "Illegal frequency" ) ;
         return -1 ;
         }
      while ((*rest == '.')  ||  ((*rest >= '0') && (*rest <= '9')))
         ++rest ;  // Skip frequency
      while (*rest == ' ')
         ++rest ;  // Skip following blanks
      n = sscanf ( rest , "%lf" , &width ) ;
      if (!n || (width <= 0.0) || (width >= 0.5)) {  // If no width specified
         strcpy ( error , "Illegal width" ) ;
         return -1 ;
         }
      while ((*rest == ' ') || (*rest == '.') || ((*rest>='0') && (*rest<='9')))
         ++rest ;  // Skip width
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < 2) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            if (id == ID_PRED_QMF)
               fprintf(fp, "\nQuadrature-mirror filter of %s to create:", rest);
            else if (id == ID_PRED_MORLET)
               fprintf(fp , "\nMorlet wavelet filter of %s to create:", rest);
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (i >= 4)
                  break ;
               switch (i) {
                  case 0:
                     fprintf ( fp , "\n  REAL : " ) ;  break ;
                  case 1:
                     fprintf ( fp , "\n  IMAG : " ) ;  break ;
                  case 2:
                     fprintf ( fp , "\n  AMP: " ) ;  break ;
                  case 3:
                     fprintf ( fp , "\n  PHASE: " ) ;  break ;
                  }
               if (misc_params.names->len[i])
                  fprintf ( fp , "%s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "{skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (id == ID_PRED_QMF) {
         if (qmf_sig ( &misc_params , sigptr , freq , width ,
                       &n_signals , &signals, error ))
            return -1 ;
         }
      else if (id == ID_PRED_MORLET) {
         if (morlet ( &misc_params , sigptr , freq , width ,
                      &n_signals , &signals, error ))
            return -1 ;
         }
      return 0 ;
      }

   if (id == ID_PRED_PADDING) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "Must be MEAN or DETREND" ) ;
         return -1 ;
         }
      if (! strcmp ( rest , "MEAN" ))
         misc_params.padding = 0 ;
      else if (! strcmp ( rest , "DETREND" ))
         misc_params.padding = 1 ;
      else {
         strcpy ( error , "Must be MEAN or DETREND" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , "\nFilter padding = %s", rest ) ;
            fclose ( fp ) ;
            }
         }
      return 0 ;
      }

   if (id == ID_PRED_MOV_AVG) {
      if ((! rest)  ||  (strlen (rest) == 0)) {
         strcpy ( error , "No signal name specified" ) ;
         return -1 ;
         }
      while (*rest == ' ')
         ++rest ;
      n = sscanf ( rest , "%d" , &m ) ;
      if (! n  ||  (m < 2)) {
         strcpy ( error , "Illegal period" ) ;
         return -1 ;
         }
      while ((*rest >= '0') && (*rest <= '9'))
         ++rest ;  // Skip period
      while (*rest == ' ')
         ++rest ;  // Skip following blanks
      for (i=0 ; i<n_signals ; i++) {  // Look for source signal
         if (! strcmp ( signals[i]->name , rest )) {
            sigptr = signals[i] ;
            break ;
            }
         }
      if (i == n_signals) {
         sprintf ( error , "Signal '%s' does not exist", rest ) ;
         return -1 ;
         }
      if (sigptr->n < m+2) {
         sprintf ( error , "Signal '%s' is too short", rest ) ;
         return -1 ;
         }
      if ((misc_params.names == NULL)  ||  ! misc_params.names->nreal) {
         strcpy ( error , "No created signal name specified" ) ;
         return -1 ;
         }
      if (strlen ( audit_log )) {
         if ((fp = fopen ( audit_log , "at" )) != NULL) {
            fprintf ( fp , " moving average %s (period=%d) to create:",
                      rest, m ) ;
            for (i=0 ; i<misc_params.names->n ; i++) {
               if (misc_params.names->len[i])
                  fprintf ( fp , "\n  %s", misc_params.names->start[i] ) ;
               else 
                  fprintf ( fp , "\n  {skip field}" ) ;
               }
            fclose ( fp ) ;
            }
         }
      if (mov_avg ( &misc_params , sigptr , m ,
                    &n_signals , &signals, error ))
         return -1 ;
      return 0 ;
      }

    return 0 ;
}

/*
--------------------------------------------------------------------------------

   Called by external routines for access to the signals and other info

--------------------------------------------------------------------------------
*/

int get_signals ( Signal ***sigs )
{
   *sigs = signals ;
   return n_signals ;
}

int get_networks ( Network ***nets )
{
   *nets = networks ;
   return n_networks ;
}

int get_ARMAs ( ARMA ***ARMs , int *fixed )
{
   *ARMs = ARMAs ;
   *fixed = arma_fixed ;
   return n_ARMAs ;
}

int get_orthogs ( Orthog ***orths )
{
   *orths = orthogs ;
   return n_orthogs ;
}

int get_ins_outs ( InputOutput ***inout )
{
   *inout = ins_outs ;
   return n_ins_outs ;
}

int get_conf_comps ( ConfComp **cc )
{
   *cc = conf_comps ;
   return n_conf_comps ;
}

void get_net_params ( int *netmodel , int *nhid1 , int *nhid2 , int *domain ,
                      int *linear , int *kernel )
{
   *netmodel = net_params.net_model ;
   *nhid1 = net_params.n_hidden1 ;
   *nhid2 = net_params.n_hidden2 ;
   *domain = net_params.domain ;
   *linear = net_params.linear ;
   *kernel = net_params.kernel ;
}

MiscParams *get_misc_params ()
{
   return &misc_params ;
}

LearnParams *get_learn_params ()
{
   return &learn_params ;
}

/*
--------------------------------------------------------------------------------

   Local routine to clear all objects

--------------------------------------------------------------------------------
*/

static void clear_all ()
{
   int i ;

   if (misc_params.names != NULL) {
      MEMTEXT ( "Names" ) ;
      delete ( misc_params.names ) ;
      misc_params.names = NULL ;
      }
   if (n_signals) {
      MEMTEXT ( "Signals" ) ;
      for (i=0 ; i<n_signals ; i++) {
         if (signals[i] != NULL)
            delete ( signals[i] ) ;
         }
      MEMTEXT ( "signal array" ) ;
      FREE ( signals ) ;
      signals = NULL ;
      n_signals = 0 ;
      }
   clear_io ( 0 , &n_ins_outs , &ins_outs ) ;
   clear_io ( 1 , &n_ins_outs , &ins_outs ) ;
   if (n_classes) {
      MEMTEXT ( "classes" ) ;
      for (i=0 ; i<n_classes ; i++) {
         if (classes[i] != NULL)
            FREE ( classes[i] ) ;
         }
      FREE ( classes ) ;
      classes = NULL ;
      n_classes = 0 ;
      }
   if (trnset != NULL) {
      MEMTEXT ( "trnset" ) ;
      delete trnset ;
      trnset = NULL ;
      }
   if (testset != NULL) {
      MEMTEXT ( "testset" ) ;
      delete testset ;
      testset = NULL ;
      }
   if (n_networks) {
      MEMTEXT ( "Networks" ) ;
      for (i=0 ; i<n_networks ; i++) {
         if (networks[i] != NULL)
            delete ( networks[i] ) ;
         }
      MEMTEXT ( "network array" ) ;
      FREE ( networks ) ;
      networks = NULL ;
      n_networks = 0 ;
      }
   if (n_ARMAs) {
      MEMTEXT ( "ARMAs" ) ;
      for (i=0 ; i<n_ARMAs ; i++) {
         if (ARMAs[i] != NULL)
            delete ( ARMAs[i] ) ;
         }
      MEMTEXT ( "ARMA array" ) ;
      FREE ( ARMAs ) ;
      ARMAs = NULL ;
      n_ARMAs = 0 ;
      }
   if (n_orthogs) {
      MEMTEXT ( "orthogs" ) ;
      for (i=0 ; i<n_orthogs ; i++) {
         if (orthogs[i] != NULL)
            delete ( orthogs[i] ) ;
         }
      MEMTEXT ( "orthogs array" ) ;
      FREE ( orthogs ) ;
      orthogs = NULL ;
      n_orthogs = 0 ;
      }
   if (conf_comps != NULL) {
      MEMTEXT ( "conf_comps" ) ;
      FREE ( conf_comps ) ;
      conf_comps = NULL ;
      n_conf_comps = 0 ;
      }
}

/*
--------------------------------------------------------------------------------

   Local routine to alphabetize the class list and simultaneously
   update the training and test sets.

--------------------------------------------------------------------------------
*/

static int alphabetize ()
{
   int i, j, nin, next, *map ;
   unsigned *nper ;
   double *dptr, *priors ;
   char **temp ;

   if ((n_classes <= 1)  ||  ((trnset == NULL)  &&  (testset == NULL)))
      return 0 ;

   MEMTEXT ( "PROCESS: alphabetize starting" ) ;
   map = (int *) MALLOC ( n_classes * sizeof(int) ) ;
   nper = (unsigned *) MALLOC ( n_classes * sizeof(unsigned) ) ;
   priors = (double *) MALLOC ( n_classes * sizeof(double) ) ;
   temp = (char **) MALLOC ( n_classes * sizeof(char *) ) ;
   if ((map == NULL) || (temp == NULL) || (nper == NULL) || (priors == NULL)) {
      if (map != NULL)
         FREE ( map ) ;
      if (temp != NULL)
         FREE ( temp ) ;
      if (nper != NULL)
         FREE ( nper ) ;
      if (priors != NULL)
         FREE ( priors ) ;
      return 1 ;
      }

   for (i=0 ; i<n_classes ; i++) {     // Selection sort
      next = -1 ;
      for (j=0 ; j<n_classes ; j++) {  // Search for lexicographic first
         if (classes[j] == NULL)       // This flags names already selected
            continue ;
         if ((next < 0)  ||  (strcmp ( classes[j] , classes[next] ) < 0))
            next = j ;                 // Keep track of smallest
         }
      temp[i] = classes[next] ;        // This is the next alphabetically
      classes[next] = NULL ;           // Flag that it is taken
      map[next] = i ;                  // This is for remapping tsets
      }

   if ((trnset != NULL)  &&  (trnset->output_mode == OUTMOD_CLASSIFICATION)) {
      for (i=0 ; i<n_classes ; i++)
         nper[i] = trnset->nper[i] ;
      for (i=0 ; i<n_classes ; i++)
         trnset->nper[i] = nper[map[i]] ;
      for (i=0 ; i<n_classes ; i++)
         priors[i] = trnset->priors[i] ;
      for (i=0 ; i<n_classes ; i++)
         trnset->priors[i] = priors[map[i]] ;
      dptr = trnset->data ;
      nin = trnset->n_inputs ;
      for (i=0 ; i<trnset->ntrain ; i++) {
         j = (int) dptr[nin] - 1 ;   // class is stored after inputs, org 1
         if (j < 0)                  // Do not remap reject category
            continue ;
         dptr[nin] = map[j] + 1.01 ; // Add a fraction for correct truncation
         dptr += trnset->size ;
         }
      }

   if ((testset != NULL)  &&  (testset->output_mode == OUTMOD_CLASSIFICATION)) {
      for (i=0 ; i<n_classes ; i++)
         nper[i] = testset->nper[i] ;
      for (i=0 ; i<n_classes ; i++)
         testset->nper[i] = nper[map[i]] ;
      for (i=0 ; i<n_classes ; i++)
         priors[i] = testset->priors[i] ;
      for (i=0 ; i<n_classes ; i++)
         testset->priors[i] = priors[map[i]] ;
      dptr = testset->data ;
      nin = testset->n_inputs ;
      for (i=0 ; i<testset->ntrain ; i++) {
         j = (int) dptr[nin] - 1 ;   // class is stored after inputs, org 1
         if (j < 0)                  // Do not remap reject category
            continue ;
         dptr[nin] = map[j] + 1.01 ; // Add a fraction for correct truncation
         dptr += testset->size ;
         }
      }

   FREE ( map ) ;
   FREE ( nper ) ;
   FREE ( priors ) ;
   FREE ( classes ) ;
   classes = temp ;

   MEMTEXT ( "PROCESS: alphabetize done" ) ;
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   Local routine to validate training set before learning

--------------------------------------------------------------------------------
*/

static int check_training_set ( char *error )
{
   int i ;

   if ((trnset == NULL)  ||  (trnset->ntrain == 0)) {
      strcpy ( error , "Cannot TRAIN NETWORK; No training set exists." ) ;
      return -1 ;
      }
   if (! trnset->n_outputs) {
      strcpy ( error , "Cannot TRAIN NETWORK; No outputs (or all REJECT)." );
      return -1 ;
      }
   if (trnset->output_mode == OUTMOD_CLASSIFICATION) {
      if (trnset->n_outputs  !=  n_classes) {
         sprintf ( error , "%d class names exist, but training set has %d.",
                   n_classes, trnset->n_outputs ) ;
         return -1 ;
         }
      for (i=0 ; i<trnset->n_outputs ; i++) {
         if (trnset->nper[i] == 0) {
            sprintf ( error , "Class %s has no cases.", classes[i]+1 ) ;
            return -1 ;
            }
         }
      }
   return 0 ;
}


/*
--------------------------------------------------------------------------------

   Local routine to validate test set before testing

--------------------------------------------------------------------------------
*/

static int check_test_set ( char *error )
{
   int i ;

   if ((testset == NULL)  ||  (testset->ntrain == 0)) {
      strcpy ( error , "Cannot TEST NETWORK; No test set exists." ) ;
      return -1 ;
      }
   if (! testset->n_outputs) {
      strcpy ( error , "Cannot TEST NETWORK; No outputs (or all REJECT)." );
      return -1 ;
      }
   if (testset->output_mode == OUTMOD_CLASSIFICATION) {
      if (testset->n_outputs  !=  n_classes) {
         sprintf ( error , "%d class names exist, but test set has %d.",
                   n_classes, testset->n_outputs ) ;
         return -1 ;
         }
      for (i=0 ; i<testset->n_outputs ; i++) {
         if (testset->nper[i] == 0) {
            sprintf ( error , "Class %s has no cases.", classes[i] ) ;
            return -1 ;
            }
         }
      }
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   Local routine to validate network parameters prior to creating a new one

--------------------------------------------------------------------------------
*/

static int check_net_params ( char *error )
{
   if (net_params.net_model == NETMOD_MLFN) {
      if ((net_params.n_hidden1 == 0)  &&  (net_params.n_hidden2 != 0)) {
         strcpy(error, "Hidden layer two must be empty if layer one is empty.");
         return -1 ;
         }
      if ((net_params.domain == DOMAIN_COMPLEX) &&
          (net_params.out_model == OUTMOD_CLASSIFICATION)) {
         strcpy ( error, "Cannot do classification with COMPLEX models." ) ;
         return -1 ;
         }
      if ((net_params.domain != DOMAIN_REAL)  &&  net_params.n_hidden2) {
         strcpy ( error, "Cannot have two hidden layers in non-REAL models." ) ;
         return -1 ;
         }
      if ((net_params.domain != DOMAIN_REAL)  &&  net_params.n_inputs % 2) {
         strcpy ( error, "Must have even number of inputs in non-REAL models.");
         return -1 ;
         }
      if ((net_params.domain == DOMAIN_COMPLEX) && net_params.n_outputs%2) {
         strcpy ( error, "Must have even number of outputs in COMPLEX models.");
         return -1 ;
         }
      if ((net_params.domain == DOMAIN_COMPLEX_HIDDEN)  &&
                                                   ! net_params.n_hidden1) {
         strcpy ( error , "Must have hidden layer in COMPLEX HIDDEN model.");
         return -1 ;
         }
      }

   if ((net_params.net_model == NETMOD_SEPVAR)  ||
       (net_params.net_model == NETMOD_SEPCLASS)) {
      if (net_params.kernel != KERNEL_GAUSS) {
         strcpy( error , "SEPVAR and SEPCLASS can use GAUSS kernel only." ) ;
         return -1 ;
         }
      }
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   Local routine to validate learning parameters prior to training

--------------------------------------------------------------------------------
*/

static int check_learn_params ( char *error )
{
   int i ;
   double *dptr ;

   if (net_params.net_model == NETMOD_MLFN) {
      if (net_params.n_hidden1  &&
           ((learn_params.method == METHOD_REGRS_CJ) ||
            (learn_params.method == METHOD_REGRS_LM))) {
         strcpy(error, "Cannot use REGRESS when there is a hidden layer." ) ;
         return -1 ;
         }
      if ((learn_params.errtype != ERRTYPE_MSE)  &&
         ((learn_params.method == METHOD_AN1_LM)  ||
          (learn_params.method == METHOD_AN2_LM)  ||
          (learn_params.method == METHOD_REGRS_LM))) {
         strcpy ( error , "LM method valid only for MSE error." ) ;
         return -1 ;
         }
      if ((trnset->output_mode != OUTMOD_CLASSIFICATION) &&
          (learn_params.errtype == ERRTYPE_XENT)) {
         strcpy ( error , "CROSS ENTROPY valid for CLASSIFICATION only.");
         return -1 ;
         }
      }

   if ((net_params.net_model == NETMOD_PNN)  ||
        (net_params.net_model == NETMOD_SEPVAR) ||
        (net_params.net_model == NETMOD_SEPCLASS)) {
      if (learn_params.siglo > learn_params.sighi) {
         strcpy(error, "Cannot TRAIN NETWORK; SIGMA LOW exceeds SIGMA HIGH.");
         return -1 ;
         }
      if (trnset->output_mode == OUTMOD_CLASSIFICATION) {
         for (i=0 ; i<trnset->ntrain ; i++) {
            dptr = trnset->data + trnset->size * i ;
            if ((int) dptr[trnset->n_inputs]  ==  0) {
               strcpy(error, "PNN cannot have REJECT class.");
               return -1 ;
               }
            }
         }
      }
   if ((net_params.net_model == NETMOD_SEPVAR)  ||
       (net_params.net_model == NETMOD_SEPCLASS)) {
      if (net_params.kernel != KERNEL_GAUSS) {
         strcpy( error , "Cannot TRAIN NETWORK; SEP... for GAUSS only." ) ;
         return -1 ;
         }
      }
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   Local routines to verify that a training/test set is compatible
   with a trained network.

--------------------------------------------------------------------------------
*/

static int check_set_net_compatibility (
   TrainingSet *tset ,
   Network *net ,
   char *error )
{
   if (check_set_net_inputs ( tset , net , error ))
      return -1 ;
   if (check_set_net_outputs ( tset , net , error ))
      return -1 ;
   return 0 ;
}


static int check_set_net_inputs (
   TrainingSet *tset ,
   Network *net ,
   char *error )
{
   int i, net_inputs ;

   net_inputs = net->n_inputs ;
   if ((net->model == NETMOD_MLFN)  &&  (((MLFN *)net)->domain != DOMAIN_REAL))
      net_inputs *= 2 ;

   if (tset->n_inputs != net_inputs) {
      sprintf( error , "Train/test inputs (%d) not equal to net inputs (%d)",
               tset->n_inputs, net_inputs ) ;
      return -1 ;
      }

   for (i=0 ; i<net_inputs ; i++) {
      if (tset->lags[i] != net->lags[i]) {
         sprintf( error , "Input %d lag for tset (%d) not equal to net (%d)",
                  i+1, tset->lags[i], net->lags[i] ) ;
         return -1 ;
         }
      }

   return 0 ;
}


static int check_set_net_outputs (
   TrainingSet *tset ,
   Network *net ,
   char *error )
{
   int i, net_outputs ;
   double *dptr ;

   net_outputs = net->n_outputs ;
   if ((net->model == NETMOD_MLFN) && (((MLFN *)net)->domain == DOMAIN_COMPLEX))
      net_outputs *= 2 ;

   if (tset->n_outputs != net_outputs) {
      sprintf ( error, "Train/test outputs (%d) not equal to net outputs (%d)",
                tset->n_outputs, net_outputs ) ;
      return -1 ;
      }
   if (tset->output_mode != net->output_mode) {
      strcpy( error , "Train output mode not equal to net mode" ) ;
      return -1 ;
      }

   if (net->output_mode == OUTMOD_CLASSIFICATION) {
      for (i=0 ; i<net->n_outputs ; i++) {
         if (strcmp ( classes[i] , net->classnames[i] )) {
            strcpy( error , "Class names are not identical" ) ;
            return -1 ;
            }
         }
      }

   if (net->output_mode == OUTMOD_MAPPING) {
      for (i=0 ; i<net_outputs ; i++) {
         if (tset->leads[i] != net->leads[i]) {
            sprintf(error, "Output %d lead for tset (%d) not equal to net (%d)",
                     i+1, tset->leads[i], net->leads[i] ) ;
            return -1 ;
            }
         }
      }

   if ((net->model == NETMOD_PNN)  ||
        (net->model == NETMOD_SEPVAR) ||
        (net->model == NETMOD_SEPCLASS)) {
      if (tset->output_mode == OUTMOD_CLASSIFICATION) {
         for (i=0 ; i<tset->ntrain ; i++) {
            dptr = tset->data + tset->size * i ;
            if ((int) dptr[tset->n_inputs]  ==  0) {
               strcpy ( error, "PNN family cannot have REJECT class.");
               return -1 ;
               }
            }
         }
      }
   return 0 ;
}

/*
--------------------------------------------------------------------------------

   ARMA compatibility checks

--------------------------------------------------------------------------------
*/

int check_arma_compatibility ( int nweights , int nout , int fixed ,
                               ARMA *arma , char *error )
{
   int i, j, n_inputs, n_ma, nw ;

   if (nweights != arma->nw) {
      sprintf ( error , "Inputs/Outputs defines %d weights, this ARMA has %d.",
                nweights, arma->nw ) ;
      return 1 ;
      }

   if (nout != arma->nout) {
      sprintf ( error , "Inputs/Outputs defines %d outputs, this ARMA has %d.",
                nout, arma->nout ) ;
      return 1 ;
      }

   if (fixed != arma->fixed) {
      strcpy ( error , "ARMA FIXED conflict" ) ;
      return 1 ;
      }

   n_inputs = n_ma = nw = 0 ;

	for (i=0 ; i<n_ins_outs ; i++) {  // Check all ins and outs
	   if (ins_outs[i]->is_input) {   // If this is an input
		   n_inputs +=	ins_outs[i]->maxlag - ins_outs[i]->minlag+1;
         for (j=ins_outs[i]->minlag ; j<=ins_outs[i]->maxlag ; j++) {
            if (! arma->is_input[nw]) {
               sprintf ( error , "Weight %d: IO is input, ARMA is MA", nw ) ;
               return 1 ;
               }
            if (arma->lag[nw] != j) {
               sprintf ( error , "Weight %d has lag conflict (IO=%d, ARMA=%d)",
                         nw, j, arma->lag[nw] ) ;
               return 1 ;
               }
            ++nw ;
            }
         }
		else {                         // This is an output
         for (j=ins_outs[i]->minlag ; j<=ins_outs[i]->maxlag ; j++) {
            if (! j)       // Only lags are MA terms
               continue ;
            if (arma->is_input[nw]) {
               sprintf ( error , "Weight %d: IO is MA, ARMA is input", nw ) ;
               return 1 ;
               }
            if (arma->lag[nw] != j) {
               sprintf ( error , "Weight %d has lag conflict (IO=%d, ARMA=%d)",
                         nw, j, arma->lag[nw] ) ;
               return 1 ;
               }
            ++n_ma ;
            ++nw ;
            }
         }
      }

   if (n_inputs != arma->nin) {
      sprintf ( error , "Inputs/Outputs defines %d inputs, this ARMA has %d.",
                n_inputs, arma->nin ) ;
      return 1 ;
      }

   if (n_ma != arma->nma) {
      sprintf ( error , "Inputs/Outputs defines %d MA, this ARMA has %d.",
                n_ma, arma->nma ) ;
      return 1 ;
      }

   return 0 ;
}
