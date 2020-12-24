/******************************************************************************/
/*                                                                            */
/*  CLASSES - Headers for all classes and structures                          */
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

#if !defined(CLASSES_H)
#define CLASSES_H

/*
   Forward declarations
*/

class Strings ;
class Signal ;


/*
--------------------------------------------------------------------------------

   Structures

--------------------------------------------------------------------------------
*/

enum RandomDensity {
   NormalDensity ,
   CauchyDensity
   } ;

enum SignalType {
  DataSignal ,             // The usual (and default) case
  CorrelationSignal ,      // Auto or cross correlation
  SpectrumSignal ,         // Spectrum
  SpectrumDevSignal        // Spectrum deviation from flattness
  } ;

struct ControlData {
   int n ;                 // Length of control buffer
   int next ;              // Offset to next command in it
   char *buf ;             // Control buffer
   } ;

struct NetParams {
   int net_model ;         // Network model (see NETMOD_? in CONST.H)
   int n_inputs ;          // Number of input neurons
	int n_outputs ;         // Number of output neurons
   int out_model ;         // Output model (see OUTMOD_? in CONST.H)
   char **classnames ;     // The n_outputs class names are here
   int kernel ;            // Parzen kernel (see KERNEL_? in CONST.H)
   int domain ;            // Layer domains for MLFN (see DOMAIN_?)
   int linear ;            // MLFN linear outputs?
	int n_hidden1 ;         // Number of hidden neurons (first hidden layer)
	int n_hidden2 ;         // Number of hidden neurons (second hidden layer)
   } ;

/*
   The annealing parameters have two values for each parameter.
   The ones with an I suffix are for starting weights.
   The ones with an E suffix are for local minimum escape.
*/

struct AnnealParams {
   int ntempI ;     // Number of temperatures
   int niterI ;     // Iterations at each temperature
   int sbI ;        // Iteration counter setback
   double startI ;  // Starting temperature
   double endI ;    // Ending temperature
   double ratioI ;  // This times std is starting prob factor
   enum RandomDensity randomI ; // NormalDensity or CauchyDensity
   int climbI ;     // Always accept improvement?
   int reducI ;     // Temp reduction schedule (EXPONENTIAL or FAST)
   int ntempE ;
   int niterE ;
   int sbE ;
   double startE ;
   double endE ;
   double ratioE ;
   enum RandomDensity randomE ;
   int climbE ;
   int reducE ;
   } ;

struct LearnParams {
// These parameters are universal
   int progress ;      // Print detailed progress info? (DOS only)
   double quit_err ;   // Quit if error this low
   int errtype ;       // Error type (ERRTYPE_?) for MLFN
   int acc ;           // Digits accuracy during retry loop
   int refine ;        // Additional digits for refinement
// These are for PNN family only
   double siglo ;      // Minimum sigma for global optimization
   double sighi ;      // And maximum
   int nsigs ;         // Number to try in that range
// These are for MLFN family only
   int method ;        // MLFN Learning method (METHOD_? in CONST.H)
   int retries ;       // Quit after this many additional tries
   int pretries ;      // Number of tries before first refinement
   struct AnnealParams *ap ;
   } ;

struct MiscParams {
   Strings *names ;        // Assorted usage for NAMES names
   int classif_output ;    // Current class for CLASSIFY training
   double threshold ;      // CLASSIFY confusion reject cutoff
   double classif_prior ;  // Prior probability weight
   int include ;           // Include first cases from training/test cumulate?
   int exclude ;           // Exclude first cases from training/test cumulate?
   int display_range ;     // 0=optimal, 1=symmetric, 2=supplied
   double display_min ;    // If specified, this is the minimum
   double display_max ;    // And this is the maximum
   int display_domain0 ;   // Minimum ordinal point to display
   int display_domain1 ;   // And Max
   double display_rate ;   // Points per unit time
   double display_origin ; // Time value corresponding to first point
   int orthog_type ;       // Orthogonalization (1=princo, 2=cent princo, 3=dis)
   int orthog_nfac ;       // Orthogonalization number of factors
   double orthog_lim ;     // Orthogonalization fraction of variance (0-1)
   int orthog_std ;        // Orthogonalization: Standardize?
   int spectrum_window ;   // 0=none, 1=Welch
   double conf_prob ;      // Confidence interval two-tailed probability
   int padding ;           // Filter padding: 0=mean, 1=detrend
   } ;

struct InputOutput {
   int is_input ;          // Is this an input (versus output)?
   int which ;             // Index in signal array
   int minlag, maxlag ;    // Lag range (Lead for outputs)
   int ordinal ;           // Counts ordinal position (in and out separately)
   int is_other ;          // If also output (input), that ordinal position
   int shock ;             // For ARMA output: shock signal #, else -1
   } ;

struct TestNetResults {
   double mse ;            // Mean squared error
   double stderr_mse ;     // Estimated std of mse
   double mae ;            // Mean absolute error
   double stderr_mae ;     // Estimated std of mae
   double p5 ;             // 5'th percentile
   double p10 ;            // Et cetera
   double p25 ;
   double p75 ;
   double p90 ;
   double p95 ;
   double abs90 ;          // 90'th percentile absolute error
   double jackbias ;       // Jackknife bias
   double jackstd ;        // and standard error
   double bootbias ;       // Bootstrap bias
   double bootstd ;        // and standard error
   } ;

enum ConfCompType {
   CCcenter ,
   CCdetrend ,
   CCoffset ,
   CCscale ,
   CCstandardize ,
   CCdifference ,
   CCseasonal ,
   CClog
   } ;

struct ConfComp {
   enum ConfCompType which ;
   int signum ;       // Ordinal number of signal to compensate
   Signal *persig ;   // Pointer to signal supplying params
   int ord_num ;      // Ordinal number (org 0) of ConfComps for this signal
   } ;


/*
--------------------------------------------------------------------------------

   Strings - multiple strings

--------------------------------------------------------------------------------
*/

class Strings {

public:
   Strings ( char *all ) ;
   ~Strings () ;
   int is_duplication () ;

   int n ;         // Number of strings, including zero length
   int nreal ;     // Number having nonzero length
   char **start ;  // String i starts at start[i]
   int *len ;      // And its length is len[i]

private:
   char *data ;
} ;


/*
--------------------------------------------------------------------------------

   SingularValueDecomp - Singular value decomposition

   The following steps are needed to compute a least-squares solution
   to a (possibly overdetermined) linear system:
     1) Create a SingularValueDecomp object.  The constructor will allocate
        memory for the design matrix 'a', the right-hand-side 'b', and all
        scratch memory that it needs.  Optionally, the user can flag the
        constructor to preserve 'a' and return the decomposition in 'u'.
        Normally, 'a' is overwritten.
     2) The design matrix must be placed in 'a' and svdcmp called.
     3) Place the right-hand-side in 'b'
     4) Allocate a vector where the solution is to be placed.
        Call backsub with a pointer to this vector.

--------------------------------------------------------------------------------
*/

class SingularValueDecomp {

public:

   SingularValueDecomp ( int nrows , int ncols , int save_a=0 ) ;
   ~SingularValueDecomp () ;
   void svdcmp () ;
   void backsub ( double limit , double *soln ) ;

   int ok ;         // Was everything legal and allocs successful?

/*
   These are made public to allow access if desired.
   Normally, only 'a' (the design matrix) and 'b' (the right-hand-side)
   are written by the user.  If 'save_a' is nonzero, 'a' is kept intact.
*/

   double *a ;      // nrows by ncols input of design, output of U
   double *u ;      // unless save_a nonzero, in which case U output in 'u'
   double *w ;      // Unsorted ncols vector of singular values
   double *v ;      // Ncols by ncols output of 'v'
   double *b ;      // Nrows right-hand-side for backsub


private:

   void bidiag ( double *matrix ) ;
   double bid1 ( int col , double *matrix , double scale ) ;
   double bid2 ( int col , double *matrix , double scale ) ;
   void right ( double *matrix ) ;
   void left ( double *matrix ) ;
   void cancel ( int low , int high , double *matrix ) ;
   void qr ( int low , int high , double *matrix ) ;
   void qr_mrot ( int col , double sine , double cosine , double *matrix ) ;
   void qr_vrot ( int col , double sine , double cosine ) ;

   int rows ;       // Nrows preserved here
   int cols ;       // And ncols
   double *work ;   // Scratch vector ncols long
   double norm ;    // Norm of 'a' matrix
} ;


/*
--------------------------------------------------------------------------------

   FFT - Mixed-radix Fast Fourier Transform

--------------------------------------------------------------------------------
*/

class FFT {

public:
   FFT ( int ndim , int spacing , int n_segments ) ;      // Constructor
   ~FFT () ;                                              // Destructor
   void cpx ( double *real , double *imag , int isign ) ; // Complex array
   void rv ( double *real , double *imag ) ;              // Real vector
   void irv ( double *real , double *imag ) ;             // Inverse real
   int ok ;

private:
   int npts ;
   int nspan ;
   int ntot ;
   int n_facs ;
   int n_sq_facs ;
   int max_factor ;
   int all_factors[64] ;
   double *rwork ;
   int *iwork ;
} ;


/*
--------------------------------------------------------------------------------

   Filter

--------------------------------------------------------------------------------
*/

class Filter {

public:
   Filter ( int npts , double *series , int pad=0 , int use_qmf=0 ) ;
   ~Filter () ;
   void lowpass ( double freq , double width , double *out ) ;
   void highpass ( double freq , double width , double *out ) ;
   void bandpass ( double freq , double width , double *out ) ;
   void morlet ( double freq , double width , double *real , double *imag ) ;
   void qmf ( double freq , double width , double *real , double *imag ) ;
   int ok ;

private:
   int length ;       // Number of points in input real series
   int n ;            // Length pushed to power of 2
   int halfn ;        // n/2
   double slope, intercept ; // For pre-detrending
   double *xr, *xi ;  // Transform here, n/2 each with xr[n/2] in xr[0]
   double *workr, *worki ; // Work areas for filters, each n long
   FFT *fft ;         // FFT object saved here
   FFT *qmf_fft ;     // If needed for quadrature-mirror filters
} ;


/*
--------------------------------------------------------------------------------

   Signal

--------------------------------------------------------------------------------
*/

class Signal {

public:
   Signal ( char *signame , int length , double *signal ) ;
   ~Signal () ;

   enum SignalType type ;  // Optimizes display
   int n ;             // Length of signal
   char *name ;        // Its name
   double *sig ;       // It is here
   int centered ;      // Has it been centered?
   double cent ;       // Its center if so
   int detrended ;     // Has it been detrended?
   double slope ;      // Slope if so
   double pivot ;      // And pivot point
   int offsetted ;     // Has it been offset?
   double offst ;      // Offset if so
   int scaled ;        // Has it been scaled?
   double scal ;       // Its scale if so
   int standardized ;  // Has it been standardized?
   double mean ;       // Its mean if so
   double std ;        // And its standard deviation
   int differenced ;   // Has it been differenced?  (Degree if so)
   double leads[3] ;   // First points before differencing
   int seasonal ;      // Has it been seasonally differenced? (Period if so)
   double *sleads ;    // First period points if so
   int source_n ;      // If CORRELATION or SPECTRUM, length of defining signal
   int known_n ;       // Normally n, else set by NET_PRED.CPP, ARMAPRED.CPP
   int npred ;         // Number of future prediction confidence intervals
   double *intervals ; // This is them (2 * npred long), alloced in NP_CONF.CPP
   int mult_conf ;     // Normally 0, 1 if 'intervals' are multiplicative

   void replace ( int n , int n_protect , double *x ) ;
   void center () ;
   int median_center () ;
   void undo_center ( Signal *per ) ;
   void detrend () ;
   void undo_detrend ( Signal *per ) ;
   void offset ( double val ) ;
   void undo_offset ( Signal *per ) ;
   void scale ( double val ) ;
   void undo_scale ( Signal *per ) ;
   void standardize () ;
   void undo_standardize ( Signal *per ) ;
   void difference ( int degree ) ;
   int undo_difference ( Signal *per ) ;
   int seasonal_diff ( int period ) ;
   int undo_seasonal_diff ( Signal *per ) ;
   void integrate ( int period ) ;
   void siglog () ;
   void sigexp () ;
} ;


/*
--------------------------------------------------------------------------------

   TrainingSet - The cases making up a training or test set.

   The actual data is in 'data'.
   If the output mode is CLASSIFICATION, each input vector is followed
   by the ordinal number (1 through nout) of its class.
   If the output model is MAPPING, the output follows the input vector.

--------------------------------------------------------------------------------
*/

class TrainingSet {

public:

   TrainingSet ( int output_mode , int n_inputs , int n_outputs ,
                 int n_inputs_outputs , InputOutput **inputs_outputs ) ;
   ~TrainingSet () ;
   void operator= ( const TrainingSet& ) ;

   int train ( NetParams *net_params , MiscParams *misc_params ,
               int n_inputs_outputs , InputOutput **inputs_outputs ,
               Signal **signals ) ;

   unsigned ntrain ; // Number of samples in 'data'
   double *data ;    // Actual training data here
   int output_mode ; // Output mode (OUTMOD_? in CONST.H)
   int n_inputs ;    // Number of inputs
   int n_outputs ;   // Number of outputs
   unsigned size ;   // Size of each case (number of doubles)
   unsigned *lags ;  // Lag of each input
   unsigned *leads ; // Lead of each output
   unsigned *nper ;  // N of cases in each class
   double *priors ;  // Prior probability weights
} ;


/*
--------------------------------------------------------------------------------

   PrincoData and Discrim

--------------------------------------------------------------------------------
*/

class PrincoData {

public:
   PrincoData ( int ncases , int nvars , double *data , int stdize ,
                int maxfacs , double frac , int *nfacs ) ;
   PrincoData ( FILE *fp , int *errnum ) ;
   ~PrincoData () ;
   void factors ( int ncases , double *data ) ;
   void eigen ( double **vals , double **vect ) ;
   int save ( FILE *fp ) ;

private:
   int nvars, nfacs ;
   double *means, *evals, *evect, *work1 ;
} ;

class Discrim {

public:
   Discrim ( int ncases , int nvars , int nclasses , double *data ,
      int *classes , int maxfacs , double frac , int *nfacs ) ;
   Discrim ( FILE *fp , int *errnum ) ;
   ~Discrim () ;
   void factors ( int ncases , double *data ) ;
   void eigen ( double **vals , double **vect ) ;
   int save ( FILE *fp ) ;

private:
   int nvars, nclasses, nfacs ;
   double *means, *evals, *evect, *work1 ;
} ;


/*
--------------------------------------------------------------------------------

   Orthog

--------------------------------------------------------------------------------
*/

class Orthog {

public:

   Orthog ( char *name , TrainingSet *tptr , MiscParams *misc ) ;
   Orthog ( char *name , int nin , int nfacs , int type , unsigned *lags ) ;
 
   ~Orthog () ;

   int transform ( int nio , InputOutput **inouts , int *nsigs ,
                   Signal ***signals , MiscParams *misc ) ;
   char name[256] ;  // Name for this model
   int nin ;         // Number of inputs
   int nfacs ;       // Number of factors computed
   int type ;        // Orthogonalization type (1-3)
   unsigned *lags ;  // Lags recorded here for safety checks
   int ok ;          // Was all constructor memory allocation successful?

   PrincoData *princo ;
   Discrim *discrim ;
} ;


/*
--------------------------------------------------------------------------------

   ARMA

--------------------------------------------------------------------------------
*/

class ARMA {

public:

   ARMA ( char *name , int nin , int nma , int nout ,
          int nio , InputOutput **inouts , int arma_fixed ) ;
   ~ARMA () ;

   int learn ( int nio , InputOutput **inouts , Signal **signals ,
               LearnParams *lptr ) ;
   void get_outputs ( int nio , InputOutput **inouts , Signal **signals ,
                      int ncases , int flag_end , int from_start ,
                      int get_shocks , double *outvars ) ;
   double get_shocks ( int nio , InputOutput **inouts , Signal **signals ,
                       int ncases , double *outvars ) ;
   double predict ( int nio , InputOutput **inouts , Signal **signals ,
                    int ncases , int shocks_known , double *outvars ) ;
   char name[256] ;  // Name for this model
   int nin ;         // Number of inputs (not including MA)
   int nma ;         // Number of MA terms
   int nout ;        // Number of outputs
   int maxlag ;      // Maximum input lag (MA ignored)
   int fixed ;       // Are output means fixed at zero?
   int nw ;          // Number of weights (length of next 2 arrays, nin+nma)
   int nvars ;       // Total params per output = nw + (fixed == 0)
   int *is_input ;   // Is this term an input (versus MA)
   int *lag ;        // Its lag (positive)
   double *var ;     // Variance vector (nout long)
   double *wts ;     // Weight matrix (nout by (nw+1), last is constant)
   double error ;    // Error variance (all outputs pooled)
   int trained ;     // Has it been trained yet?
   int nio ;         // Length of next array
   InputOutput *ins_outs ; // For save/restore: audit log can print this info
   int ok ;          // Was all constructor memory allocation successful?
} ;


/*
--------------------------------------------------------------------------------

   Network - Pure virtual base class for all networks

--------------------------------------------------------------------------------
*/

class Network {

public:

   Network ( char *name , NetParams *net_params ) ;
   virtual ~Network () = 0 ;

   virtual int trial ( double *input ) = 0 ;
   virtual double trial_error ( TrainingSet *tptr ) = 0 ;
   virtual int learn ( TrainingSet *tptr , LearnParams *lptr ) = 0 ;
   virtual int wt_save ( FILE *fp ) = 0 ;
   virtual int wt_restore ( FILE *fp ) = 0 ;
   virtual int wt_print ( char *name ) = 0 ;
   int testnet ( TrainingSet *tptr , double threshold , int *confusion ,
                 double *confuse , int extended , TestNetResults *res ) ;
   char name[256] ;  // Name for this network
   int n_inputs ;    // Number of input neurons
   int n_outputs ;   // Number of output neurons
   int model ;       // Network model (NETMOD_? in CONST.H)
   int output_mode ; // Output mode (OUTMOD_? in CONST.H)
   char **classnames ; // The n_outputs class names are here
   unsigned *lags ;  // Lag of each input
   unsigned *leads ; // Lead of each output
   double *out ;     // Outputs computed from an input
   double neterr ;   // Mean square error of the network if executed
   int errtype ;     // Network error definition (ERRTYPE_?) used to train it
   int ok ;          // Was all constructor memory allocation successful?

protected:
   NetParams netparams ; // The parameters that created this network
} ;

/*
--------------------------------------------------------------------------------

   NPconf - Nonparametric confidence intervals

--------------------------------------------------------------------------------
*/

class NPconf {

public:
   NPconf ( int nvars , int npred , int ncases , double prob ,
            int n_conf_comps , ConfComp *conf_comps ) ;
   ~NPconf () ;
   int sig_init ( int ivar , int isig , Signal *sig ) ;
   void insert ( int ivar , int idist , double xknown , double xpred ) ;
   void eval ( int casenum , int offset ) ;
   void conf ( double *excess_tail_prob , double *cover_prob ) ;

   int ok ;

private:
   int nv ;
   int np ;
   int nc ;
   double prob ;
   double **intervals ;
   double *x ;
   double *known ;
   double *pred ;
   double **ccwork ;
   int *isigs ;
   Signal **sigs ;
   int ncc ;
   ConfComp *ccs ;
} ;


/*
--------------------------------------------------------------------------------

   MLFN

   Nhid1 and nhid2 are the number of neurons in the first and second layers
   respectively.  Either or both may be zero.  If nhid1 is zero, it is assumed
   that nhid2 is also zero.

   Weights for each layer are stored as a vector.
   The first weight in hid1_coefs is the weight from the first input
   to the first neuron in the first hidden layer.

   In addition to n_inputs, nhid1, nhid2 and n_outputs, we keep nin_n, nhid1_n,
   nhid2_n and nout_n.  These are the actual number of numbers going OUT
   of the layer, including the bias term.
   For network domain REAL they will each be equal to their partner, plus 1.
   For the COMPLEX models, some of them will be double their partner, plus 2.
   The only exception is that nout_n is just nout times 1 or 2, since the
   output layer does not have a bias term feeding a following layer!
   The number of elements not including the bias term is in the _w form.

   The output activation is the identity function if outlin is nonzero.
   Otherwise it is the same nonlinear function as the hiddens neurons.

--------------------------------------------------------------------------------
*/

class MLFN: public Network {

public:

   MLFN ( char *name , NetParams *net_params , int zero = 1 ) ;
   ~MLFN () ;
   int trial ( double *input ) ;
   double trial_error ( TrainingSet *tptr ) ;
   int learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int wt_save ( FILE *fp ) ;
   int wt_restore ( FILE *fp ) ;
   int wt_print ( char *name ) ;

   int anx ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int ssg ( TrainingSet *tptr , struct LearnParams *lptr , int use_grad ) ;
   double regress ( TrainingSet *tptr , SingularValueDecomp *sptr ) ;
   double gradient ( TrainingSet *tptr , double *work1 ,
                     double *work2 , double *grad ) ;
         void errderiv_r ( int nout , int iout , double *outs ,
                           double target , double *err , double *deriv ) ;
         void errderiv_c ( int nout , int iout , double *outs ,
                           double target_r , double target_i ,
                           double *err , double *deriv ) ;
         double gradient_complex ( TrainingSet *tptr , double *work1 ,
                                   double *work2 , double *grad ) ;
         double gradient_real ( TrainingSet *tptr , double *work1 ,
                                double *work2 , double *grad ) ;
   int anx_dd ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   double lm_core ( TrainingSet *tptr , double *work1 ,
                    double *work2 , double *alpha , double *beta ) ;
   double lm_core_real ( TrainingSet *tptr , double *hid2delta ,
                         double *grad , double *alpha , double *beta ) ;
      void process_real ( double *input , int idep , double target ,
                          double *err , double *alpha , double *beta ,
                          double *hid2delta , double *grad );
      double lm_core_complex ( TrainingSet *tptr , double *work1 ,
                               double *grad , double *alpha , double *beta ) ;
      void process_cr ( double *input , int idep , double target ,
                        double *err , double *alpha , double *beta ,
                        double *work1 , double *grad ) ;
      void process_cc ( double *input , int idep , double target , int ipart ,
                        double *err , double *alpha , double *beta ,
                        double *work1 , double *grad ) ;
   int regrs_dd ( TrainingSet *tptr , struct LearnParams *lptr ) ;

   int nhid1 ;      // Number of neurons in first hidden layer
   int nhid2 ;      // And in second hidden layer

   int nin_n ;      // For REAL models, these will equal the above, plus 1.
   int nhid1_n ;    // For COMPLEX models, some of these are double the above
   int nhid2_n ;    // plus 2, because they are the number of numbers going out
   int nout_n ;     // except nout_n does not include bias term

   int nin_w ;      // For REAL models, these will be nin.
   int nhid1_w ;    // For COMPLEX models, some of these are double.
   int nhid2_w ;

   int ntot ;       // Total number of weights (numbers)
   int nhid ;       // Of which this many are for hidden neurons

   double *all_weights ; // Weight vector (ntot numbers) starts here

   int domain ;     // REAL, COMPLEX etc. (DOMAIN_? in CONST.H)
   int outlin ;     // Outputs linear (identity activation function)?

private:
   double *hid1_coefs ; // nhid1 * nin_n weights (in changes fastest)
   double *hid2_coefs ; // nhid2 * nhid1_n weights (hid1 changes fastest)
   double *out_coefs ;  // nout * nhid?_n weights (hid? changes fastest)
   double *hid1 ;       // First hidden layer activations
   double *hid2 ;       // And second
} ;


/*
--------------------------------------------------------------------------------

   PNNet - Derived from Network, this is a virtual base class for all PNNs

--------------------------------------------------------------------------------
*/

class PNNet: public Network {

public:

   PNNet ( char *name , NetParams *net_params ) ;
   virtual ~PNNet () ;

   virtual int trial ( double *input ) = 0 ;
   virtual int trial_deriv ( double *input , int tclass , double *target ) = 0 ;
   double trial_error ( TrainingSet *tptr ) ;
   double trial_error ( TrainingSet *tptr , int find_deriv ) ;
   virtual int learn ( TrainingSet *tptr , struct LearnParams *lptr ) = 0 ;

   double *deriv ;  // Computed derivative
   double *deriv2 ; // Computed second derivative
   virtual int wt_save ( FILE *fp ) = 0 ;
   virtual int wt_restore ( FILE *fp ) = 0 ;
   virtual int wt_print ( char *name ) = 0 ;

   int kernel ;     // Parzen kernel (KERNEL_? in CONST.H)

protected:
   TrainingSet *tdata ;  // Training data for classification is here
} ;

/*
--------------------------------------------------------------------------------

   PNNbasic - Basic version of PNNet

--------------------------------------------------------------------------------
*/

class PNNbasic: public PNNet {

public:
   PNNbasic ( char *name , NetParams *net_params ) ;
   ~PNNbasic () ;

   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ;
   int learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int wt_save ( FILE *fp ) ;
   int wt_restore ( FILE *fp ) ;
   int wt_print ( char *name ) ;
   double sigma ;
} ;


/*
--------------------------------------------------------------------------------

   PNNsepvar - Separate sigma for each variable version of PNNet

--------------------------------------------------------------------------------
*/

class PNNsepvar: public PNNet {

public:
   PNNsepvar ( char *name , NetParams *net_params ) ;
   ~PNNsepvar () ;

   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ;
   int learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int wt_save ( FILE *fp ) ;
   int wt_restore ( FILE *fp ) ;
   int wt_print ( char *name ) ;
   double *sigma ;

private:
   double *v ;      // Scratch for derivative computation allocd in 'learn'
   double *w ;      // Ditto
   double *dsqr ;   // Ditto
} ;


/*
--------------------------------------------------------------------------------

   PNNsepclass - Separate sigma for each variable and class version of PNNet

--------------------------------------------------------------------------------
*/

class PNNsepclass: public PNNet {

public:
   PNNsepclass ( char *name , NetParams *net_params ) ;
   ~PNNsepclass () ;

   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ;
   int learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int wt_save ( FILE *fp ) ;
   int wt_restore ( FILE *fp ) ;
   int wt_print ( char *name ) ;
   double *sigma ;

private:
   double *v ;      // Scratch for derivative computation allocd in 'learn'
   double *w ;      // Ditto
   double *dsqr ;   // Ditto
} ;

#endif
