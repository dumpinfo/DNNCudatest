/******************************************************************************/
/*                                                                            */
/* CONST.H - System and program limitation constants                          */
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

#if !defined(CONST_H)
#define CONST_H

#define MAXPOSNUM 2147483647

#if ! defined ( PI )
#define PI 3.141592653589793
#endif

#define MISSING -1.0  /* Returned by ParseDouble for missing data */

/*
   These control memory allocation interception for diagnostics
*/

#if 1
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

/*
	These are command id codes.  Commands are parsed and the appropriate code
   is generated.  That code is then passed to another routine for processing.
   This complexity is not needed for simple DOS implementations, but it helps
   for more complex GUI versions.
*/

#define ID_PRED_INITIALIZE 1000
#define ID_PRED_AUDIT_LOG 1001
#define ID_PRED_BYE 1002
#define ID_PRED_CLEAR_ALL 1003
#define ID_PRED_PROGRESS_ON 1004
#define ID_PRED_PROGRESS_OFF 1005
#define ID_PRED_COMMANDFILE 1006
#define ID_PRED_NAME 1007
#define ID_PRED_READSIG 1008
#define ID_PRED_SAVESIG 1009
#define ID_PRED_DISPLAY 1010
#define ID_PRED_DISPLAY_RANGE 1011
#define ID_PRED_DISPLAY_DOMAIN 1012
#define ID_PRED_DISPLAY_RATE 1013
#define ID_PRED_DISPLAY_ORIGIN 1014
#define ID_PRED_DISPLAY_CONFIDENCE 1015
#define ID_PRED_CLEAR_INPUT 1016
#define ID_PRED_CLEAR_OUTPUT 1017
#define ID_PRED_INPUT 1018
#define ID_PRED_OUTPUT 1019
#define ID_PRED_CLEAR_CLASSES 1020
#define ID_PRED_CLASS 1021
#define ID_PRED_PRIOR 1022
#define ID_PRED_CLEAR_TRAINING 1023
#define ID_PRED_CUMULATE_TRAINING 1024
#define ID_PRED_CLEAR_TEST 1025
#define ID_PRED_CUMULATE_TEST 1026
#define ID_PRED_CUMULATE_INCLUDE 1027
#define ID_PRED_CUMULATE_EXCLUDE 1028
#define ID_PRED_NETMOD 1029
#define ID_PRED_KERNEL 1030
#define ID_PRED_MLFN_HID1 1031
#define ID_PRED_MLFN_HID2 1032
#define ID_PRED_MLFN_OUTPUT_ACT 1033
#define ID_PRED_MLFN_DOMAIN 1034
#define ID_PRED_QUITERR 1035
#define ID_PRED_ACCURACY 1036
#define ID_PRED_REFINE 1037
#define ID_PRED_SIGLO 1038
#define ID_PRED_SIGHI 1039
#define ID_PRED_SIGTRIES 1040
#define ID_PRED_MLFN_ERROR_TYPE 1041
#define ID_PRED_MLFN_LEARNING 1042
#define ID_PRED_MLFN_RESTARTS 1043
#define ID_PRED_MLFN_PRETRIES 1044
#define ID_PRED_ANNEAL_INIT_TEMPS 1045
#define ID_PRED_ANNEAL_ESCAPE_TEMPS 1046
#define ID_PRED_ANNEAL_INIT_ITERS 1047
#define ID_PRED_ANNEAL_ESCAPE_ITERS 1048
#define ID_PRED_ANNEAL_INIT_SETBACK 1049
#define ID_PRED_ANNEAL_ESCAPE_SETBACK 1050
#define ID_PRED_ANNEAL_INIT_START 1051
#define ID_PRED_ANNEAL_ESCAPE_START 1052
#define ID_PRED_ANNEAL_INIT_STOP 1053
#define ID_PRED_ANNEAL_ESCAPE_STOP 1054
#define ID_PRED_ANNEAL_INIT_RATIO 1055
#define ID_PRED_ANNEAL_ESCAPE_RATIO 1056
#define ID_PRED_ANNEAL_INIT_REDUCTION 1057
#define ID_PRED_ANNEAL_ESCAPE_REDUCTION 1058
#define ID_PRED_ANNEAL_INIT_RANDOM 1059
#define ID_PRED_ANNEAL_ESCAPE_RANDOM 1060
#define ID_PRED_ANNEAL_INIT_ACCEPT 1061
#define ID_PRED_ANNEAL_ESCAPE_ACCEPT 1062
#define ID_PRED_CLEAR_NETWORKS 1063
#define ID_PRED_CLEAR_NETWORK 1064
#define ID_PRED_TRAIN_NETWORK 1065
#define ID_PRED_CVTRAIN_NETWORK 1066
#define ID_PRED_SAVE_NETWORK 1067
#define ID_PRED_RESTORE_NETWORK 1068
#define ID_PRED_PRINT_NETWORK 1069
#define ID_PRED_THRESHOLD 1070
#define ID_PRED_TEST_NETWORK 1071
#define ID_PRED_EXTENDED_TEST_NETWORK 1072
#define ID_PRED_GENERATE 1073
#define ID_PRED_NETWORK_PREDICT 1074
#define ID_PRED_NETWORK_CONFIDENCE 1075
#define ID_PRED_CLEAR_ARMAS 1076
#define ID_PRED_CLEAR_ARMA 1077
#define ID_PRED_TRAIN_ARMA 1078
#define ID_PRED_ARMA_SHOCK 1079
#define ID_PRED_ARMA_FIXED 1080
#define ID_PRED_ARMA_PREDICT 1081
#define ID_PRED_ARMA_CONFIDENCE 1082
#define ID_PRED_SAVE_ARMA 1083
#define ID_PRED_RESTORE_ARMA 1084
#define ID_PRED_CENTER 1085
#define ID_PRED_MEDIAN_CENTER 1086
#define ID_PRED_UNDO_CENTER 1087
#define ID_PRED_DETREND 1088
#define ID_PRED_UNDO_DETREND 1089
#define ID_PRED_OFFSET 1090
#define ID_PRED_UNDO_OFFSET 1091
#define ID_PRED_SCALE 1092
#define ID_PRED_UNDO_SCALE 1093
#define ID_PRED_STANDARDIZE 1094
#define ID_PRED_UNDO_STANDARDIZE 1095
#define ID_PRED_DIFFERENCE 1096
#define ID_PRED_UNDO_DIFFERENCE 1097
#define ID_PRED_SEASONAL_DIFFERENCE 1098
#define ID_PRED_UNDO_SEASONAL_DIFFERENCE 1099
#define ID_PRED_LOG 1100
#define ID_PRED_EXP 1101
#define ID_PRED_CONFIDENCE_CENTER 1102
#define ID_PRED_CONFIDENCE_DETREND 1103
#define ID_PRED_CONFIDENCE_OFFSET 1104
#define ID_PRED_CONFIDENCE_SCALE 1105
#define ID_PRED_CONFIDENCE_STANDARDIZE 1106
#define ID_PRED_CONFIDENCE_DIFFERENCE 1107
#define ID_PRED_CONFIDENCE_SEASONAL 1108
#define ID_PRED_CONFIDENCE_LOG 1109
#define ID_PRED_CLEAR_CONFIDENCE 1110
#define ID_PRED_CONFIDENCE_PROB 1111
#define ID_PRED_ADD 1112
#define ID_PRED_SUBTRACT 1113
#define ID_PRED_MULTIPLY 1114
#define ID_PRED_DIVIDE 1115
#define ID_PRED_ORTHOG_TYPE 1116
#define ID_PRED_ORTHOG_FACTORS 1117
#define ID_PRED_ORTHOG_LIMIT 1118
#define ID_PRED_ORTHOG_STD 1119
#define ID_PRED_DEFINE_ORTHOG 1120
#define ID_PRED_APPLY_ORTHOG 1121
#define ID_PRED_SAVE_ORTHOG 1122
#define ID_PRED_RESTORE_ORTHOG 1123
#define ID_PRED_CLEAR_ORTHOG 1124
#define ID_PRED_CLEAR_ORTHOGS 1125
#define ID_PRED_AUTOCORR 1126
#define ID_PRED_CROSSCORR 1127
#define ID_PRED_P_AUTOCORR 1128
#define ID_PRED_P_CROSSCORR 1129
#define ID_PRED_COPY 1130
#define ID_PRED_INTEGRATE 1131
#define ID_PRED_SPECTRUM 1132
#define ID_PRED_SPECTRUM_WINDOW 1133
#define ID_PRED_SAVGOL 1134
#define ID_PRED_MAXENT 1135
#define ID_PRED_LOWPASS 1136
#define ID_PRED_HIGHPASS 1137
#define ID_PRED_BANDPASS 1138
#define ID_PRED_QMF 1139
#define ID_PRED_MORLET 1140
#define ID_PRED_PADDING 1141
#define ID_PRED_MOV_AVG 1142

/*
   These are output model codes.  If additional outputs are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.
*/

#define OUTMOD_CLASSIFICATION 1
#define OUTMOD_MAPPING 2

/*
   These are network model codes.  If additional networks are defined, they
   should be appended, leaving existing ones with the same codes, in order
   to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define NETMOD_PNN 1
#define NETMOD_SEPVAR 2
#define NETMOD_SEPCLASS 3
#define NETMOD_MLFN 4

/*
   These are parzen kernel codes.  If additional kernels are defined, they
   should be appended, leaving existing ones with the same codes, in order
	to avoid disk file incompatibilites.  They must be positive (-1 = unset).
*/

#define KERNEL_GAUSS 1
#define KERNEL_RECIP 2

/*
   These are domains for MLFN models
*/

#define DOMAIN_REAL 1
#define DOMAIN_COMPLEX 2
#define DOMAIN_COMPLEX_INPUT 3
#define DOMAIN_COMPLEX_HIDDEN 4

/*
   These are network error definitions.  They must be positive.
*/

#define ERRTYPE_MSE 1
#define ERRTYPE_ABS 2
#define ERRTYPE_KK 3
#define ERRTYPE_XENT 4

/*
   These define the learning algorithm to be used
*/

#define METHOD_AN1 1
#define METHOD_AN1_CJ 2
#define METHOD_AN2 3
#define METHOD_AN2_CJ 4
#define METHOD_AN1_LM 5
#define METHOD_AN2_LM 6
#define METHOD_REGRS_CJ 7
#define METHOD_REGRS_LM 8
#define METHOD_SS 9
#define METHOD_SSG 10

/*
   These are for assorted learning parameters
*/

#define ANNEAL_REDUCE_EXPONENTIAL 1
#define ANNEAL_REDUCE_FAST 2

#define NEURON_ON 1.0
#define NEURON_OFF 0.0

#endif
