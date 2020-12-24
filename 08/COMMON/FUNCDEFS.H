#if !defined(FUNCDEFS_H)
#define FUNCDEFS_H

extern double actderiv ( double f ) ;
extern void activity_rr ( double *input , double *coefs , double *output ,
                          int ninputs , int linear ) ;
extern void activity_cr ( double *input , double *coefs , double *output ,
                          int ninputs , int linear ) ;
extern void activity_cc ( double *input , double *coefs , double *output ,
                          int ninputs , int linear ) ;
extern double act_func ( double x ) ;
extern void act_func_init () ;
extern double anneal1 ( int n , double *x , double *work ,
                        double (*criter) ( double * ) , double bestfval ,
                        int itry , int ntemps , int niters , int setback ,
                        double starttemp , double stoptemp ,
                        enum RandomDensity density, double fquit, int progress);
extern double anneal2 ( int n , double *x , double *work , double *work2 ,
                        double (*criter) ( double * ) , double bestfval ,
                        int ntemps , int niters , int setback ,
                        double starttemp , double stoptemp ,
                        enum RandomDensity density , double ratio , int climb ,
                        int reduction , double fquit , int progress ) ;
extern int append_io ( int is_input , char *rest , 
                       int n_signals , Signal **signals ,
                       int *nio , InputOutput ***inputs_outputs ) ;
extern void append_message ( char *msg ) ;
extern int armaconf ( int npred , ARMA *arma , int n_inputs_outputs ,
                      InputOutput **in_out , Signal **signals ,
                      int n_conf_comps , ConfComp *conf_comps ,
                      double prob , double *excess , double *toler ) ;
extern int armapred ( int npred , ARMA *arma , 
                      MiscParams *misc , int n_inputs_outputs ,
                      InputOutput **in_out , int *nsigs , Signal ***signals ) ;
extern ARMA *arma_restore ( char *armaname , char *filename , int *errnum ) ;
extern int arma_save ( ARMA *arma , char *filename ) ;
extern int autocorr ( MiscParams *misc , int operation , int ncases ,
                      Signal *sig1 , Signal *sig2 , int *nsigs ,
                      Signal ***signals , char *error , double **corrs ) ;
extern void best_graphlab ( double dmin , double dmax , int minticks ,
                            int maxticks , double *gmin , double *gmax ,
                            double *dif , int *ntot , int *nfrac ) ;
extern double beta ( int v1 , int v2 ) ;
extern void burg ( int n , double *x , int maxlag , double *pcorr ,
                   double *coefs , double *prev , double *alpha , double *beta);
extern void cauchy ( int n , double scale , double *x ) ;
extern double brentmin ( int itmax , double critlim , double eps ,
                         double tol , double (*criter) (double) ,
                         double *x1 , double *x2 , double *x3 ,
                         double y , int progress ) ;
extern void clear_io ( int is_input , int *nio , InputOutput ***inputs_outputs);
extern void close_textmode () ;
extern int combine ( MiscParams *misc , int operation , Signal *sig1 ,
                Signal *sig2 , int *nsigs , Signal ***signals , char *error ) ;
extern double conjgrad (   int itmax , double critlim , double tol,
                           double (*criter) (double * , int , double * ) ,
                           int n , double *x , double ystart ,
                           double *base , double *direc , double *g ,
                           double *h , int progress ) ;
extern int copy ( MiscParams *misc , int ncases , Signal *sig ,
                  int *nsigs , Signal ***signals , char *error ) ;
extern int cvtrain ( TrainingSet *tptr , Network *net , struct NetParams *np ,
                     struct LearnParams *lptr , double *cverror ) ;
extern void defaults ( NetParams *net_params , LearnParams *learn_params ,
                       MiscParams *misc_params ) ;
extern double dermin ( int itmax , double critlim , double tol ,
                  double (*criter) (double * , int , double * , double * ) ,
                  int n , double *x , double y , double *base , double *direc ,
                  double *g , double *h , double *dwk2 , int progress ) ;
extern void destroy_progress_window () ;
extern void display ( Signal *sptr , MiscParams *misc , int id ) ;
extern double dotprod ( int n , double *vec1 , double *vec2 ) ;
extern void dotprodc ( int n , double *vec1 , double *vec2 ,
                       double *re , double *im ) ;
extern double dotprodcr ( int n , double *vec1 , double *vec2 ) ;
extern void drawline ( int x1 , int y1 , int x2 , int y2 , int color ) ;
extern void error_message ( const char *msg ) ;
extern void exit_graphics () ;
extern int filt_sig ( MiscParams *misc , int id , Signal *sig ,
                      double freq, double width,
                      int *nsigs , Signal ***signals , char *error ) ;
extern long flrand () ;
extern long flrandmax () ;
extern int generate ( MiscParams *misc , char *rest , int *nsigs ,
                      Signal ***signals , char *error ) ;
extern int get_ARMAs ( ARMA ***ARMs , int *fixed ) ;
extern int get_conf_comps ( ConfComp ***cc ) ;
extern void get_control_line ( ControlData *cbuf , char *line ) ;
extern int get_ins_outs ( InputOutput ***inout ) ;
extern LearnParams *get_learn_params () ;
extern MiscParams *get_misc_params () ;
extern int get_networks ( Network ***nets ) ;
extern void get_net_params ( int *netmodel , int *nhid1 , int *nhid2 ,
                             int *domain , int *linear , int *kernel ) ;
extern int get_orthogs ( Orthog ***orths ) ;
extern int get_signals ( Signal ***sigs ) ;
extern int glob_min ( double low , double high , int npts , int log_space ,
	double critlim , double (*criter) (double) ,
	double *x1, double *y1 , double *x2, double *y2 ,
   double *x3, double *y3 , int progress ) ;
extern void goto_graphics ( int *nrows , int *ncols , int *chrows ,
                            int *chcols , double *aspect_ratio ) ;
extern void graphlab ( double dmin , double dmax , int nticks ,
                       double *gmin , double *gmax , double *dif ,
                       int *ntot , int *nfrac ) ;
extern int init_graphics () ;
extern void init_textmode () ;
extern int interpret_control_line ( char *line , char **rest ) ;
extern double inverse_act ( double f ) ;
extern void inverse_act_cc ( double *out , double *net ) ;
extern double lev_marq ( int itmax , double critlim , double tol ,
                         double (*criter) (double * , double * , double * ) ,
                         int nvars , double *x ,
                         SingularValueDecomp *sptr , double *grad ,
                         double *delta , double *hessian , int progress ) ;
extern void limit ( int nvars , double *x , double lim ) ;
extern void make_progress_window ( char *title ) ;
extern int maxent ( MiscParams *misc , int ncases , int degree , Signal *sig ,
                    int *nsigs , Signal ***signals , char *error ) ;
extern void *memalloc ( unsigned int n ) ;
extern void memclose () ;
extern void memfree ( void *ptr ) ;
extern void *memrealloc ( void *ptr , unsigned int size ) ;
extern void memtext ( char *text ) ;
extern int morlet ( MiscParams *misc , Signal *sig ,
                    double freq , double width ,
                    int *nsigs , Signal ***signals , char *error ) ;
extern int mov_avg ( MiscParams *misc , Signal *sig , int period ,
                     int *nsigs , Signal ***signals , char *error ) ;
extern int net_conf ( int n_inputs , int npred , Network *net , 
                      MiscParams *misc , int n_inputs_outputs ,
                      InputOutput **in_out , int nsigs , Signal **signals ,
                      int n_conf_comps , ConfComp *conf_comps ,
                      double *excess , double *toler ) ;
extern int net_pred ( int n_inputs , int npred , Network *net , 
                      MiscParams *misc , int n_inputs_outputs ,
                      InputOutput **in_out , int *nsigs , Signal ***signals ) ;
extern double normal () ;
extern void normal_pair ( double *x1 , double *x2 ) ;
extern void nomemclose () ;
extern void notext ( char *text ) ;
extern Orthog *orth_restore ( char *orthname , char *filename , int *errnum ) ;
extern int orth_save ( Orthog *orth , char *filename ) ;
extern double ParseDouble ( char **str ) ;
extern void partial_cc ( double *input , double *coefs ,
                         double *output , int ninputs ,
                         double *deriv_rr , double *deriv_ri ,
                         double *deriv_ir , double *deriv_ii , int linear ) ;
extern void partial_cr ( double *input , double *coefs , double *output ,
                         int ninputs , double *deriv_re , double *deriv_im ,
                         int linear ) ;
extern double powell ( int maxits , double critlim , double tol ,
                       double (*criter) ( double * ) , int n , double *x ,
                       double ystart , double *base , double *p0 ,
                       double *direc , int progress ) ;
extern int process ( int id , char *rest , ControlData *cbuf , char *error ,
                     char *results ) ;
extern double p_limit ( int n , int m , double limit ) ;
extern int qmf_sig ( MiscParams *misc , Signal *sig ,
                     double freq , double width ,
                     int *nsigs , Signal ***signals , char *error ) ;
extern void qsort ( int first , int last , double data[] ) ;
extern void rand_sphere ( int nvars , double *x ) ;
extern int readsig ( MiscParams *misc , char *namelist , int *nsigs ,
                     Signal ***signals , char *error ) ;
extern void remove_display ( Signal *sig ) ;
extern void restore_screen () ;
extern int save_screen () ;
extern int savgol ( MiscParams *misc , int ncases , int degree , Signal *sig ,
                    int *nsigs , Signal ***signals , char *error ) ;
extern void sflrand ( long iseed ) ;
extern void shake ( int nvars , double *center , double *x , double temp ,
                    enum RandomDensity dens ) ;
extern double shift ( double *grad , double *pert , double weight , int n ) ;
extern int sig_save ( Signal *sig , char *filename ) ;
extern int spectrum ( MiscParams *misc , Signal *sig ,int *nsigs ,
              Signal ***signals , double *dmax , double *alpha , char *error ) ;
extern double ssg_core ( int n , double *x ,
                      double (*criter)( double * , double , int , double * ) ,
                      double bestfval , int ntemps , int niters , int setback ,
                      double starttemp , double stoptemp ,
                      enum RandomDensity density , double fquit , int use_grad ,
                      double *avg , double *best , double *grad ,
                      double *avg_grad , int progress ) ;
void str_to_upr ( char *str ) ;
double t_limit ( int n , int m , double limit ) ;
extern double unifrand () ;
extern int user_pressed_escape () ;
extern void write_graphics_text ( int row , int col , char *text , int color ) ;
extern void write_progress ( char *text ) ;
extern void write_non_progress ( char *text ) ;
extern Network *wt_restore ( char *netname , char *filename , int *errnum ) ;
extern int wt_save ( Network *net , char *name ) ;

#endif
