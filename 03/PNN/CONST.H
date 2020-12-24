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
/* CONST.H - System and program limitation constants                          */
/*           This also contains typedefs, structs, et cetera.                 */
/*                                                                            */
/* The #if above MALLOC controls whether or not the diagnostic memory         */
/* allocation routines are used.  They only slow things a tiny bit.           */
/*                                                                            */
/* RANDMAX may be system dependent.  See your documentation.                  */
/*                                                                            */
/* TRAIN_BUF_SIZE makes training set memory allocation faster by avoiding     */
/* many calls to realloc.  Users of 32 bit machines with much memory may      */
/* want to increase this considerably.                                        */
/*                                                                            */
/******************************************************************************/

/*
   These interpose memory allocation checking.
*/

#if 0
#define MALLOC memalloc
#define FREE memfree
#define REALLOC memrealloc
#define MEMTEXT memtext
#define MEMCLOSE memclose
#else
#define MALLOC malloc
#define FREE free
#define REALLOC realloc
#define MEMTEXT notext
#define MEMCLOSE nomemclose
#endif

#define KEY_ESCAPE 27

#define VERSION_16_BIT 0

#if VERSION_16_BIT
#define TRAIN_BUF_SIZE 16384     /* Alloc this much tsets mem (max) at a time */
#else
#define TRAIN_BUF_SIZE 65536
#endif

#define RANDMAX 32767            /* rand() returns from 0 through this */

/*
   These are network model codes.  If additional networks are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define NETMOD_BASIC 1
#define NETMOD_SEPVAR 2
#define NETMOD_SEPCLASS 3
#define NETMOD_GCNN 4
#define NETMOD_GCNN_EW 5

/*
   These are parzen kernel codes.  If additional kernels are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define KERNEL_GAUSS 1
#define KERNEL_RECIP 2

/*
   These are output model codes.  If additional outputs are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define OUTMOD_CLASSIFICATION 1
#define OUTMOD_AUTOASSOCIATION 2
#define OUTMOD_MAPPING 3

/*
   These are command id codes.  Commands are parsed and the appropriate code
   is generated.  That code is then passed to another routine for processing.
   This complexity is not needed for simple DOS implementations, but it helps
   for more complex GUI versions.
*/

#define ID_PNNCOM_BYE 1001
#define ID_PNNCOM_COMMANDFILE 1002
#define ID_PNNCOM_NETMOD 1003
#define ID_PNNCOM_KERNEL 1004
#define ID_PNNCOM_OUTMOD 1005
#define ID_PNNCOM_NIN 1006
#define ID_PNNCOM_NOUT 1007
#define ID_PNNCOM_TRAIN 1008
#define ID_PNNCOM_QUITERR 1009
#define ID_PNNCOM_QUITTOL 1010
#define ID_PNNCOM_SIGLO 1011
#define ID_PNNCOM_SIGHI 1012
#define ID_PNNCOM_SIGTRIES 1013
#define ID_PNNCOM_LEARN 1014
#define ID_PNNCOM_SAVENET 1015
#define ID_PNNCOM_RESTNET 1016
#define ID_PNNCOM_CLEARTRAIN 1017
#define ID_PNNCOM_CLEARNET 1018
#define ID_PNNCOM_CLASSOUT 1019
#define ID_PNNCOM_PRIOR 1020
#define ID_PNNCOM_OUTFILE 1021
#define ID_PNNCOM_EXECUTE 1022
#define ID_PNNCOM_TEST 1023
#define ID_PNNCOM_CLASSIFY 1024
#define ID_PNNCOM_RESET 1025
#define ID_PNNCOM_THRESH 1026
#define ID_PNNCOM_SHOWCON 1027
#define ID_PNNCOM_SAVECON 1028
#define ID_PNNCOM_MAXMOM 1029

struct ControlData {
   int n ;                 // Length of control buffer
   int next ;              // Offset to next command in it
   char *buf ;             // Control buffer
   } ;

struct NetParams {
   int net_model ;         // PNNet model (see NETMOD_? in CONST.H)
   int net_kernel ;        // Parzen kernel (see KERNEL_? in CONST.H)
   int maxmom ;            // GCNN max moment
   int out_model ;         // Output model (see OUTMOD_? in CONST.H)
   int n_inputs ;          // Number of input neurons
   int n_outputs ;         // Number of output neurons
   } ;

struct LearnParams {
   double quit_err ;   // Quit if error this low
   double quit_tol ;   // Quit if relative improvement this low
   double siglo ;      // Minimum sigma for global optimization
   double sighi ;      // And maximum
   int nsigs ;         // Number to try in that range
   } ;

struct MiscParams {
   int classif_output ;    // Current class for CLASSIFY training
   char out_file[256] ;    // File for EXECUTE output
   double threshold ;      // CLASSIFY confusion reject cutoff
   double classif_prior ;  // Prior probability weight
   } ;

typedef struct {
   double mean ;
   double std ;
   double c3, c4, c5, c6, c7, cx ;  // cx is c8 for GCNN, c9 for EW
   } GCNNcoefs ;
