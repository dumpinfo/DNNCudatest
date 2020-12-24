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

extern double actderiv ( double f ) ;
extern void activity_rr ( double *input , double *coefs , double *output ,
                          int ninputs , int linear ) ;
extern void activity_cr ( double *input , double *coefs , double *output ,
                          int ninputs , int linear ) ;
extern void activity_cc ( double *input , double *coefs , double *output ,
                          int ninputs , int linear ) ;
extern double act_func ( double x ) ;
extern void act_func_init () ;
extern double beta ( int v1 , int v2 ) ;
extern void cauchy ( int n , double scale , double *x ) ;
extern double dotprod ( int n , double *vec1 , double *vec2 ) ;
extern void dotprodc ( int n , double *vec1 , double *vec2 ,
                       double *re , double *im ) ;
extern double dotprodcr ( int n , double *vec1 , double *vec2 ) ;
extern void error_message ( char *msg ) ;
extern long flrand () ;
extern long flrandmax () ;
extern void get_control_line ( char *control_line , int *control_file_number ,
                               FILE *control_files[] ) ;
extern double inverse_act ( double f ) ;
extern void inverse_act_cc ( double *out , double *net ) ;
extern void limit ( int nvars , double *x , double lim ) ;
extern long longrand () ;
extern long longrandmax () ;
extern void *memalloc ( unsigned int n ) ;
extern void memclose () ;
extern void memfree ( void *ptr ) ;
extern void *memrealloc ( void *ptr , unsigned int size ) ;
extern void memtext ( char *text ) ;
extern void memory_message ( char *msg ) ;
extern void nomemclose () ;
extern void normal_pair ( double *x1 , double *x2 ) ;
extern void notext ( char *text ) ;
extern double normal () ;
extern void normal_message ( char *msg ) ;
extern double ParseDouble ( char **str ) ;
extern void partial_cc ( double *input , double *coefs ,
                         double *output , int ninputs ,
                         double *deriv_rr , double *deriv_ri ,
                         double *deriv_ir , double *deriv_ii , int linear ) ;
extern void partial_cr ( double *input , double *coefs , double *output ,
                         int ninputs , double *deriv_re , double *deriv_im ,
                         int linear ) ;
extern void progress_message ( char *msg ) ;
extern void rand_sphere ( int nvars , double *x ) ;
extern void shake ( int nvars , double *center , double *x , double temp ,
                    enum RandomDensity dens ) ;
extern void slongrand ( long iseed ) ;
extern void split_control_line ( char *line , char **command , int *n_command ,
                                 char **rest , int *n_rest ) ;
extern void stack_control_file ( char *filename , int *control_file_number ,
                                 FILE *control_files[] ) ;
extern void str_to_upr ( char *str ) ;
extern void sflrand ( long iseed ) ;
extern double unifrand () ;
extern int user_pressed_escape () ;
extern void warning_message ( char *msg ) ;
extern LayerNet *wt_restore ( char *name ) ;
extern void wt_save ( LayerNet *net , int updating , char *name ) ;
