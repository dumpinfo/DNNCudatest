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
/*  CLASSES - Headers for all classes                                         */
/*                                                                            */
/******************************************************************************/

/*
--------------------------------------------------------------------------------

   TrainingSet - Collection of samples which will be used for training

   All training data is stored in 'data'.  If the output model is HETERO, the
   input for each sample is followed in 'data' by the number (1 through nout)
   of the output class to which that sample belongs.  If the output model is
   GENERAL, the output values follow the input values.  No output values are
   stored in AUTO mode, since the output is the input!

   The variable 'bufcnt' is the number of samples which could still fit in
   'data' without allocating more memory.  This allows us to allocate memory
   in large blocks, avoiding the overhead of often expensive operating system
   calls to malloc.

--------------------------------------------------------------------------------
*/

class TrainingSet {

public:

   TrainingSet ( int outmodel , int n_inputs , int n_outputs ) ;
   ~TrainingSet () ;
   void operator= ( const TrainingSet& ) ;

   void train ( char *file , int outclass , double prior ) ;

   unsigned ntrain ; // Number of samples in 'data'
   double *data ;    // Actual training data here
   int outmod ;      // Output model (see OUTMOD_? in CONST.H)
   int nin ;         // Number of input elements
   int nout ;        // Number of output elements
   unsigned size ;   // Size of each case (number of doubles)
   unsigned *nper ;  // N of cases in each class
   double *priors ;  // Prior probability weights

private:

   unsigned bufcnt ; // Sample areas remaining unused at end of 'data'
} ;

/*
--------------------------------------------------------------------------------

   PNNet

--------------------------------------------------------------------------------
*/

class PNNet {

public:

   PNNet (int net_kernel , int outmodel , int n_inputs , int n_outputs ) ;
   virtual ~PNNet () ;
   void classify_from_file ( char *name , double thresh ) ;
   void execute_from_file ( char *inname , char *outname ) ;
   void reset_confusion () ;
   void save_confusion ( char *name ) ;
   void show_confusion () ;
   void test_from_file ( char *inname ) ;
   double trial_error ( TrainingSet *tptr , int deriv ) ;
   virtual void learn ( TrainingSet *tptr , struct LearnParams *lptr ) = 0 ;
   virtual int privatize_tset () = 0 ;
   virtual int trial ( double *input ) = 0 ;
   virtual int trial_deriv ( double *input , int tclass , double *target ) = 0 ;
   virtual int wt_save ( FILE *fp ) = 0 ;
   virtual void wt_restore ( FILE *fp ) = 0 ;

   int nin ;        // Number of input neurons
   int nout ;       // Number of output neurons

   int model ;      // BASIC etc. (NETMOD_? in CONST.H)
   int kernel ;     // Parzen kernel (KERNEL_? in CONST.H)
   int maxmom ;     // GCNN max moment used
   int outmod ;     // Output model (see OUTMOD_? in CONST.H)
   int ok ;         // Was all constructor memory allocation successful?
   int errtype ;    // Flags if trained
   double neterr ;  // Mean square error of the network if executed
   int *confusion ; // Handy work area avoids malloc/free (see CONFUSE.CPP)
   double *out ;    // Output activations
   double *deriv ;  // Computed derivative
   double *deriv2 ; // Computed second derivative
} ;

/*
--------------------------------------------------------------------------------

   PNNbasic - Basic version of PNNet

--------------------------------------------------------------------------------
*/

class PNNbasic: public PNNet {

public:
   PNNbasic ( int net_kernel , int outmodel , int n_inputs , int n_outputs ) ;
   ~PNNbasic () ;
   void learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int privatize_tset () ;
   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ;
   int wt_save ( FILE *fp ) ;
   void wt_restore ( FILE *fp ) ;
   double sigma ;

private:
   TrainingSet *tdata ;  // Training data for classification is here
   int tdata_shared ;    // Is it still shared with tset?
} ;


/*
--------------------------------------------------------------------------------

   PNNsepvar - Separate sigma for each variable version of PNNet

--------------------------------------------------------------------------------
*/

class PNNsepvar: public PNNet {

public:
   PNNsepvar ( int net_kernel , int outmodel , int n_inputs , int n_outputs ) ;
   ~PNNsepvar () ;
   void learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int privatize_tset () ;
   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ;
   int wt_save ( FILE *fp ) ;
   void wt_restore ( FILE *fp ) ;
   double *sigma ;

private:
   TrainingSet *tdata ;  // Training data for classification is here
   int tdata_shared ;    // Is it still shared with tset?
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
   PNNsepclass ( int net_kernel , int outmodel , int n_inputs , int n_outputs ) ;
   ~PNNsepclass () ;
   void learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int privatize_tset () ;
   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ;
   int wt_save ( FILE *fp ) ;
   void wt_restore ( FILE *fp ) ;
   double *sigma ;

private:
   TrainingSet *tdata ;  // Training data for classification is here
   int tdata_shared ;    // Is it still shared with tset?
   double *v ;      // Scratch for derivative computation allocd in 'learn'
   double *w ;      // Ditto
   double *dsqr ;   // Ditto
} ;

/*
--------------------------------------------------------------------------------

   GCNN - Gram-Charlier network

--------------------------------------------------------------------------------
*/

class GCNN: public PNNet {

public:
   GCNN ( int maxmoment , int ew , int outmodel , int n_inputs , int n_outputs);
   ~GCNN () ;
   void learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   int privatize_tset () ;  // Not needed except from virtual in parent
   int trial ( double *input ) ;
   int trial_deriv ( double *input , int tclass , double *target ) ; // no need
   int wt_save ( FILE *fp ) ;
   void wt_restore ( FILE *fp ) ;

private:
   int ewmod ;       // GCNN Edgeworth modification?
   GCNNcoefs *c ;    // Coefficients for all classes (&mean, std)
   double *priors ;  // Prior probability weights
} ;
