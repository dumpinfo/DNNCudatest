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

extern void append_message ( char *msg ) ;
extern double brentmin ( int itmax , double critlim , double eps ,
                         double tol , double (*criter) (double) ,
                         double *x1 , double *x2 , double *x3 , double y ) ;
extern void defaults ( NetParams *net_params , LearnParams *learn_params ,
                       MiscParams *misc_params ) ;
extern double derivmin ( int itmax , double critlim , double eps , double tol ,
                  double (*criter) (double * , double * , double * , int ) ,
                  int n , double *x , double y , double *base , double *direc ,
                  double *dwk1 , double *dwk2 ) ;
extern double dermin ( int itmax , double critlim , double eps , double tol ,
                  double (*criter) (double * , double * , double * , int ) ,
                  int n , double *x , double y , double *base , double *direc ,
                  double *g , double *h , double *dwk2 ) ;
extern void destroy_progress_window () ;
extern double dotprod ( int n , double *vec1 , double *vec2 ) ;
extern void error_message ( char *msg ) ;
extern int get_control_line ( ControlData *cbuf , char **rest ) ;
extern int get_yn ( char *msg ) ;
extern int glob_min ( double low , double high , int npts , int log_space ,
   double critlim , double (*criter) (double) ,
   double *x1, double *y1 , double *x2, double *y2 , double *x3, double *y3 ) ;
extern int gold_min ( int log_space , double critlim , double tol ,
   double (*criter) (double) ,
   double *x1, double *y1 , double *x2, double *y2 , double *x3, double *y3 ) ;
extern void make_progress_window () ;
extern void *memalloc ( unsigned int n ) ;
extern void memclose () ;
extern void memfree ( void *ptr ) ;
extern void *memrealloc ( void *ptr , unsigned int size ) ;
extern void memtext ( char *text ) ;
extern void memory_message ( char *msg ) ;
extern void normal_message ( char *msg ) ;
extern void nomemclose () ;
extern void notext ( char *text ) ;
extern double ParseDouble ( char **str ) ;
extern int process ( int id , char *rest , NetParams *net_params ,
                     LearnParams *learn_params , MiscParams *misc_params ,
                     ControlData *cbuf , PNNet **network , TrainingSet **tset );
extern void stack_control_file ( char *filename , int *control_file_number ,
                                 FILE *control_files[] ) ;
void str_to_upr ( char *str ) ;
extern int user_pressed_escape () ;
extern void warning_message ( char *msg ) ;
extern void write_progress ( char *text ) ;
extern PNNet *wt_restore ( char *name ) ;
extern void wt_save ( PNNet *net , int updating , char *name ) ;
