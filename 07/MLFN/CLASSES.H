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

   TrainingSet ( int netmodel , int outmodel , int n_inputs , int n_outputs ) ;
   ~TrainingSet () ;
   void train ( char *file , int outclass ) ;

   unsigned ntrain ; // Number of samples in 'data'
   double *data ;    // Actual training data here
   int outmod ;      // Output model (see OUTMOD_? in CONST.H)
   int nin ;         // Number of input elements
   int nout ;        // Number of output elements

private:

   unsigned bufcnt ; // Sample areas remaining unused at end of 'data'
} ;

/*
--------------------------------------------------------------------------------

   SingularValueDecomp - Singular value decomposition of matrices

   Normally, the user would:
     1) Use 'new' to create a SingularValueDecomp object with all necessary
        memory (a, u?, w, v, work, b) allocated by the constructor.
     2) Fill in the public 'a' with the matrix to be decomposed.
     3) Call svdcmp to decompose a, replacing it with the U matrix if preserve
        is zero, else computing u.
        This will also compute w and v, which are normally not used but are
        public just in case the user wants to access them.
     4) Fill in the 'b' vector with the right hand side of the equations.
     5) Call backsub with a pointer to the cols vector which is where the
        solution will be placed.  This vector is NOT allocated by the
        constructor.  The outputs of svdcmp (a, u?, w, v) will not be disturbed.
     6) Repeat the above step as desired.
     7) Delete the SingularValueDecomp object, which frees all memory which
        was allocated by the constructor.

--------------------------------------------------------------------------------
*/

class SingularValueDecomp {

public:

   SingularValueDecomp ( int rows , int cols , int preserve ) ;
   ~SingularValueDecomp () ;
   void svdcmp () ;
   void backsub ( double thresh , double *x ) ;

   int ok ;         // Was memory allocation successful?

/*
   The following four input/output areas are allocated by the constructor
*/

   double *a ;      // Rows by cols input of 'A' matrix, output of U
   double *u ;      // unless preserve != 0, in which case U output here
   double *w ;      // Cols vector output of singular values, not sorted
   double *v ;      // Cols by cols output of 'V' matrix
   double *b ;      // Rows vector of RHS input to backsub


private:

   int rows ;       // Number of rows in 'A' matrix
   int cols ;       // and number of columns
   double *work ;   // Cols work vector (allocated by constructor)
} ;

/*
--------------------------------------------------------------------------------

   LayerNet

   Nhid1 and nhid2 are the number of neurons in the first and second layers
   respectively.  Either or both may be zero.  If nhid1 is zero, it is assumed
   that nhid2 is also zero.

   Weights for a layer are stored as a two dimensional matrix strung out into
   a vector.  For example, the first element in hid1_coefs is the weight
   connecting the first input neuron to the first hidden neuron.  The second
   connects the second input to the first hidden.  The nin+1 element is the
   bias for the first hidden unit.  The nin+2 connects the first input to the
   second hidden, et cetera.

   In addition to nin, nhid1, nhid2 and nout, we keep nin_n, nhid1_n,
   nhid2_n and nout_n.  These are the actual number of numbers going OUT
   of the layer, including the bias term.
   For network model REAL they will each be equal to their partner, plus 1.
   For the COMPLEX models, some of them will be double their partner, plus 2.
   The only exception is that nout_n is just nout times 1 or 2, since the
   output layer does not have a bias term feeding a following layer!
   The number of elements not including the bias term is in the _w form.

   The output activation is the identity function if outlin is nonzero.
   Otherwise it is the same nonlinear function as the hiddens neurons.

--------------------------------------------------------------------------------
*/

class LayerNet {

public:

   void an1 ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   void an1_cj ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   void regrs_cj ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   void classify_from_file ( char *name , double thresh ) ;
   void execute_from_file ( char *inname , char *outname ) ;
   LayerNet ( int netmodel , int outmodel , int outlinear ,int n_inputs ,
          int n_hidden1 , int n_hidden2 , int n_outputs , int exe , int zero ) ;
   ~LayerNet () ;
   void learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   void reset_confusion () ;
   void save_confusion ( char *name ) ;
   void show_confusion () ;
   void test_from_file ( char *inname ) ;
   void trial ( double *input ) ;
   double trial_error ( TrainingSet *tptr ) ;
   void wt_print ( char *name ) ;
   int wt_save ( FILE *fp ) ;
   void wt_restore ( FILE *fp ) ;

   int nin ;        // Number of input neurons
   int nhid1 ;      // Number of neurons in hidden layer 1
   int nhid2 ;      // Ditto for hidden layer 2
   int nout ;       // Number of output neurons

   int nin_n ;      // For REAL models, these will equal the above, plus 1.
   int nhid1_n ;    // For COMPLEX models, some of these are double the above
   int nhid2_n ;    // plus 2, because they are the number of numbers going out
   int nout_n ;     // except nout_n does not include bias term

   int nin_w ;      // For REAL models, these will be nin.
   int nhid1_w ;    // For COMPLEX models, some of these are 2 * nhid.
   int nhid2_w ;

   int model ;      // REAL, COMPLEX etc. (NETMOD_? in CONST.H)
   int errtype ;    // Network error definition (ERRTYPE_?) used to train it
   int outlin ;     // Outputs linear (identity activation function)?
   int exe ;        // Are work areas allocated to allow executing it?
   int ok ;         // Was all constructor memory allocation successful?
   double neterr ;  // Mean square error of the network if executed
   int outmod ;     // Output model (see OUTMOD_? in CONST.H)

private:
    
   int anneal1 ( TrainingSet *tptr , struct LearnParams *lptr ,
                 LayerNet *bestnet , int init ) ;
   double conjgrad ( TrainingSet *tptr , int maxits ,
                     double reltol , double errtol , int progress ) ;
      void check_grad ( TrainingSet *tptr , double *grad ) ;
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
      void find_new_dir ( double gam , double *g , double *h , double *grad ) ;
      double gamma ( double *g , double *grad ) ;
   void copy_weights ( LayerNet *dest , LayerNet *source ) ;
   double direcmin ( TrainingSet *tptr , double start_err ,
                     int itmax , double eps , double tol ,
                     double *base , double *direc ) ;
      void negate_dir ( double *direc ) ;
      void preserve ( double *base ) ;
      void step_out ( double step , double *direc , double *base ) ;
      void update_dir ( double step , double *direc ) ;
   int genetic ( TrainingSet *tptr , struct LearnParams *lptr ) ;
      void decode ( char *popptr , int reg ) ;
   void genetic_learn ( TrainingSet *tptr , struct LearnParams *lptr ) ;
   void perturb ( LayerNet *cent , LayerNet *perturbed , double temp ,
                  int regress) ;
   double regress ( TrainingSet *tptr , SingularValueDecomp *sptr ) ;
   void zero_weights () ;

   int *confusion ; // Handy work area avoids malloc/free (see CONFUSE.CPP)

   double *hid1_coefs ; // nhid1 * (nin+1) weights (in changes fastest)
   double *hid2_coefs ; // nhid2 * (nhid1+1) weights (hid1 changes fastest)
   double *out_coefs ;  // nout * (nhid?+1) weights (hid? changes fastest)
   double *hid1 ;       // Hid 1 neuron activations here if exe nonzero
   double *hid2 ;       // Ditto hidden layer 2
   double *out ;        // Ditto outputs
} ;
