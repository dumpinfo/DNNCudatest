/******************************************************************************/
/*                                                                            */
/*  EIGEN - Eigenstructure routines, including PrincoData and Discrim classes */
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

/*
--------------------------------------------------------------------------------

   jacobi - Eigenstructure of symmetric real matrix

--------------------------------------------------------------------------------
*/

inline void rotate ( double *mat , int i , int j , int k , int l ,
              double sine , double tau , int n )
{
   double t1, t2 ;

   t1 = mat[i*n+j] ;
   t2 = mat[k*n+l] ;
   mat[i*n+j] = t1 - sine * (t2 + t1 * tau) ;
   mat[k*n+l] = t2 + sine * (t1 - t2 * tau) ;
}

void jacobi (
   int n ,           // Size of matrix
   double *mat ,     // Square symmetric real matrix
   double *evals ,   // Output of n eigenvalues
   double *evect ,   // Output of n by n eigenvectors (each is a column)
   double *work1 ,   // Work vector n long
   double *work2     // Work vector n long
   )
{
   int i, j, k, ibig, sweep ;
   double err, *dptr, threshold, test, diff, theta ;
   double corr, sine, cosine, tangent, tau, big, mat_ij ;

/*
   The rotations will be cumulated in the evect matrix, each of whose
   columns will be a normalized eigenvector.  Initialize this to the identity.
   The output vector evals will always contain the current diagonal of mat,
   so initialize it now.  The corrections to the diagonal as a result of
   rotation will be maintained in work1 and work2, so also initialize them.
*/

   for (i=0 ; i<n ; i++) {
      dptr = evect + i * n ;  // Point to row i
      for (j=0 ; j<n ; j++)   // Set entire row to 0
         dptr[j] = 0.0 ;
      dptr[i] = 1.0 ;         // But set diagonal to 1
      evals[i] = work1[i] = mat[i*n+i] ; // Matrix diagonal
      work2[i] = 0.0 ;
      }

/*
   This is the main loop which does a single sweep through the matrix.
   It is rare that more than a dozen sweeps will be needed do zero the
   above-diagonal region.  However, just for safety, we impose a limit here.
*/

   for (sweep=0 ; sweep<200 ; sweep++) {

/*
   Start by checking for convergence.  We simply sum the magnitude of the
   elements above the diagonal.  When they become tiny we are done.
*/

      err = 0.0 ;
      for (i=0 ; i<n-1 ; i++) {
         dptr = mat + i * n ;          // Point to row i
         for (j=i+1 ; j<n ; j++)       // Above-diagonal area
            err += fabs ( dptr[j] ) ;  // Will be all zero when done
         }
      if (err < 1.e-60)  // Safer for portability than using zero
         break ;         // Although numerically slightly inferior

/*
   We can save a lot of time in the early sweeps by skipping a zeroing
   rotation for matrix elements that are already small.  Set the threshold
   slightly smaller than recommended in Press et al to be more conservative.
   After four sweeps (1 more than Press uses), use a threshold of zero
   so that we do the rotation no matter how small.
*/

      if (sweep > 4)
         threshold = 0.0 ;
      else
         threshold = 0.15 * err / (n * n) ;

/*
   This is the so-called "cyclic Jacobi" method.  We pass through the
   entire above-diagonal region in this nested loop.  For small matrices
   it is more efficient to zero the largest element each time.  But for
   large matrices the search for the largest becomes too expensive.
*/

      for (i=0 ; i<n-1 ; i++) {     // Row i
         for (j=i+1 ; j<n ; j++) {  // And column j
            mat_ij = mat[i*n+j] ;   // This is the element to be zeroed

/*
   Compute a test value as a moderate multiple of the off-diagonal element
   currently being zeroed.  Many decisions will be based on the significance
   of this value, which is determined by adding it to the appropriate
   quantity and seeing if the result is the same to machine precision.
*/

            test = 128.0 * fabs ( mat_ij ) ;

/*
   Here we cheat a little, though not until a sweep later than Press does.
   If the off-diagonal being zeroed is already tiny compared to both of
   its corresponding diagonals, we cross our fingers and surreptitiously
   just set this element to zero without actually doing the work.  Shameful,
   but it saves time with no real cost in accuracy.
*/

            if ((sweep > 5)  &&
                (fabs(evals[i]) == (fabs(evals[i])+test))  &&
                (fabs(evals[j]) == (fabs(evals[j])+test))) {
               mat[i*n+j] = 0.0 ;
               continue ;
               }

/*
   This is another time saver.  It is not cheating at all.
   On the first few iterations there is no point in bothering with zeroing
   small elements.  Stick with the big ones.  Only after we have a few sweeps
   under our belt do we get picky.  The threshold was computed earlier.
*/

            if (fabs ( mat_ij ) < threshold)
               continue ;

/*
   This is it.  We can no longer avoid it.  We must do the rotations to
   zero the (i,j) above-diagonal element.  Start by computing the rotation
   angle, theta, then find our ultimate goal, its tangent.
   If the denominator of the strict formula would be tiny, use an
   approximation that is essentially perfect in that instance.
*/

            diff = evals[j] - evals[i] ;
            if (fabs(diff) == (fabs(diff) + test)) {
               if (diff != 0.0)
                  tangent = mat_ij / diff ;
               else 
                  tangent = 0.0 ;
               }
            else {
               theta = 0.5 * diff / mat_ij ;
               tangent = 1.0 / (fabs ( theta ) + sqrt ( theta * theta + 1.0 )) ;
               if (theta < 0.0)
                  tangent = -tangent ;
               }

/*
   Do the rotations.  We break it up into three steps, as this is an
   efficient way to handle the requirement that we only do the elements
   above the diagonal.
*/

            cosine = 1.0 / sqrt ( tangent * tangent + 1.0 ) ;
            sine = cosine * tangent ;
            tau = sine / (cosine + 1.0) ;

            for (k=0 ; k<i ; k++)     // Rows above (less than) i
               rotate ( mat , k , i , k , j , sine , tau , n ) ;
            for (k=i+1 ; k<j ; k++)   // Rows between i and j
               rotate ( mat , i , k , k , j , sine , tau , n ) ;
            for (k=j+1 ; k<n ; k++)   // Rows beyond j
               rotate ( mat , i , k , j , k , sine , tau , n ) ;

/*
   Now apply this same rotation to the eigenvector matrix
*/

            for (k=0 ; k<n ; k++)
               rotate ( evect , k , i , k , j , sine , tau , n ) ;

/*
   The last step in this (i,j) zeroing is to cumulate (across the sweep)
   the terms that will be needed to update the diagonal (eigenvalues)
   at the end of the sweep.  Also, we do not actually compute the value
   of the zeroed term, as it is zero (or should be).  Just set it.
*/

            corr = tangent * mat_ij ;
            evals[i] -= corr ;
            work2[i] -= corr ;
            evals[j] += corr ;
            work2[j] += corr ;

            mat[i*n+j] = 0.0 ;
            } // j
         }// i

/*
   A sweep is completed.  Update the eigenvalues (diagonal) and work vectors.
*/

      for (k=0 ; k<n ; k++) {
         work1[k] += work2[k] ;
         evals[k] = work1[k] ;
         work2[k] = 0.0 ;
         }

      } // sweep

/*
   We are essentially done.  The final step is to sort the eigenvalues
   in descending order and simultaneously swap the vectors.
*/

   for (i=0 ; i<n ; i++) {         // End of each pass gets next biggest
      big = evals[i] ;             // Keep track of biggest eval here
      ibig = i ;                   // And its location in array
      for (j=i+1 ; j<n ; j++) {    // Check for any bigger below
         if (evals[j] > big) {     // If we find a bigger eval
            big = evals[j] ;       // Update our record
            ibig = j ;             // And its location
            }
         }
      if (ibig == i)               // If nothing bigger was found below
         continue ;                // Then no swap necessary
      evals[ibig] = evals[i] ;     // Swap this small one into big's place
      evals[i] = big ;             // And put big one up here
      for (j=0 ; j<n ; j++) {      // Also must swap corresponding vectors
         test = evect[j*n+i] ;     // All rows of this pair of columns
         evect[j*n+i] = evect[j*n+ibig] ;
         evect[j*n+ibig] = test ;
         }
      }
}

/*
--------------------------------------------------------------------------------

   eigen_bia - Eigenstructure of B-inverse * A

--------------------------------------------------------------------------------
*/

void eigen_bia (
   int n ,           // Size of matrix
   double *a ,       // Square symmetric real matrix (destroyed)
   double *b ,       // Ditto (destroyed)
   double *evals ,   // Output of n eigenvalues of (B inverse) * A
   double *evect ,   // Output of n by n eigenvectors (each is a column)
   double *work1 ,   // Work vector n long
   double *work2     // Work vector n long
   )
{
   int i, j, k ;
   double temp ;

/*
   Replace B with itself raised to the -1/2 power
*/

   jacobi ( n , b , evals , evect , work1 , work2 ) ;
   for (i=0 ; i<n ; i++) {
      if (evals[i] > 1.e-6)               // This is the usual condition
         temp = 1.0 / sqrt ( evals[i] ) ; // But avoid division by 0
      else                                // This generally signals a very
         temp = 0.0 ;                     // bad singularity condition!
      for (j=0 ; j<n ; j++)
         b[j*n+i] = evect[j*n+i] * temp ;
      }

/*
   Premultiply A by the transpose of this matrix, and postmultiply by it
*/

   for (i=0 ; i<n ; i++) {                  // Compute (B inverse)' * A
      for (j=0 ; j<n ; j++) {
         temp = 0.0 ;
         for (k=0 ; k<n ; k++)
            temp += b[k*n+i] * a[k*n+j] ;
         evect[i*n+j] = temp ;              // Temporarily save product here
         }
      }

   for (i=0 ; i<n ; i++) {                  // Postmutiply by B
      for (j=0 ; j<n ; j++) {
         temp = 0.0 ;
         for (k=0 ; k<n ; k++)
            temp += evect[i*n+k] * b[k*n+j] ;
         a[i*n+j] = temp ;
         }
      }

/*
   Find the eigenstructure of the matrix just computed, then compute the
   final eigenvectors by premultiplying by the matrix in B.
*/

   jacobi ( n , a , evals , evect , work1 , work2 ) ;

   for (i=0 ; i<n ; i++) {       // Premultiply eigenvectors by B
      for (j=0 ; j<n ; j++) {
         temp = 0.0 ;
         for (k=0 ; k<n ; k++)
            temp += b[i*n+k] * evect[k*n+j] ;
         a[i*n+j] = temp ;       // Temporarily save eigenvectors here
         }
      }

/*
   Normalize the column eigenvectors to unit length
*/

   for (i=0 ; i<n ; i++) {       // Column i is an eigenvector
      temp = 0.0 ;               // Will cumulate squared length here
      for (j=0 ; j<n ; j++)      // Run down the rows of this column
         temp += a[j*n+i] * a[j*n+i] ; // Cumulating squared length
      if (temp > 0.0)            // Should always be true
         temp = 1.0 / sqrt ( temp ) ; // But take no chances
      for (j=0 ; j<n ; j++)      // Run down the rows of this column
         evect[j*n+i] = a[j*n+i] * temp ;  // Normalizing to unit length
      }
}


/*
--------------------------------------------------------------------------------

   PrincoData - Compute principal components of data array

--------------------------------------------------------------------------------
*/

PrincoData::PrincoData (
   int ncases ,      // Number of cases (rows in data array)
   int nv ,          // Number of variables (columns in data array)
   double *data ,    // Input (ncases by nv) matrix
   int stdize ,      // Standardize to equal variance?
   int maxfacs ,     // Maximum number of output factors
   double frac ,     // Fraction (0-1) of variance to retain
   int *nfactors     // Output: Number of factors retained
   )
{
   int n, icase, var, var2, fac, npos ;
   double *dptr, *std, diff1, diff2, sum, temp, *work2, *covar ;

/*
   Copy parameters to private area and allocate memory.
   Also realize that the number of factors cannot exceed one less than
   the number of cases.  This becomes important when we are using group
   means rather than raw data to define the factors.
*/

   MEMTEXT ( "EIGEN: PrincoData constructor" ) ;

   if (maxfacs > ncases-1)
      maxfacs = ncases-1 ;

   nvars = nv ;

   if (stdize) {
      std = (double *) MALLOC ( nvars * sizeof(double) ) ;
      if (std == NULL) {
         *nfactors = nfacs = -1 ; // Error flag
         return ;
         }
      }
   else
      std = NULL ;

   means = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evals = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evect = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   covar = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   work1 = (double *) MALLOC ( nvars * sizeof(double) ) ;
   work2 = (double *) MALLOC ( nvars * sizeof(double) ) ;

   if ((means == NULL)  ||  (evals == NULL)  ||  (evect == NULL)  ||
       (covar == NULL)  ||  (work1 == NULL)  ||  (work2 == NULL)) {
      if (means != NULL)
         FREE ( means ) ;
      if (evals != NULL)
         FREE ( evals ) ;
      if (evect != NULL)
         FREE ( evect ) ;
      if (std != NULL)
         FREE ( std ) ;
      if (covar != NULL)
         FREE ( covar ) ;
      if (work1 != NULL)
         FREE ( work1 ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      *nfactors = nfacs = -1 ;  // Error flag
      return ;
      }

/*
   Compute means and covariances
*/

   for (var=0 ; var<nvars ; var++)  // Zero the mean vector
      means[var] = 0.0 ;

   n = nvars * nvars ;              // Zero the covariance matrix
   while (n--)
      covar[n] = 0.0 ;

   for (icase=0 ; icase<ncases ; icase++) { // Cumulate means
      dptr = data + icase * nvars ;         // Point to this case
      for (var=0 ; var<nvars ; var++)       // All variables in this case
         means[var] += dptr[var] ;
      }

   for (var=0 ; var<nvars ; var++)          // Mean is sum divided by n
      means[var] /= ncases ;

   for (icase=0 ; icase<ncases ; icase++) { // Cumulate covariances
      dptr = data + icase * nvars ;         // Point to this case
      for (var=0 ; var<nvars ; var++) {
         diff1 = dptr[var] - means[var] ;
         for (var2=var ; var2<nvars ; var2++) {
            diff2 = dptr[var2] - means[var2] ;
            covar[var*nvars+var2] += diff1 * diff2 ;
            }
         }
      }

   for (var=0 ; var<nvars ; var++) {         // Divide sums by n
      for (var2=var ; var2<nvars ; var2++) { // To get covariance matrix
         covar[var*nvars+var2] /= ncases ;
         if (var != var2)                    // Symmetrically duplicate
            covar[var2*nvars+var] = covar[var*nvars+var2] ;
         }
      }

/*
   If we are to standardize, save the standard deviations and adjust
   the covariance matrix.
*/

   if (std != NULL) {
      for (var=0 ; var<nvars ; var++) {             // Each variable's variance
         std[var] = sqrt ( covar[var*nvars+var] ) + 1.e-30 ; // is in diagonal
         covar[var*nvars+var] = 1.0 ;               // which then becomes unity
         }
      for (var=0 ; var<nvars-1 ; var++) {           // Adjust covariances
         for (var2=var+1 ; var2<nvars ; var2++) {   // for standardization
            covar[var*nvars+var2] /= (std[var] * std[var2]) ;
            covar[var2*nvars+var] = covar[var*nvars+var2] ;
            }
         }
      }

/*
   Compute eigenstructure and number of factors to retain
*/

   jacobi ( nvars , covar , evals , evect , work1 , work2 ) ;

   sum = 0.0 ;
   for (var=0 ; var<nvars ; var++)
      sum += evals[var] ;       // Sum the eigenvalues

   frac *= sum ;                // User wants this much of sum retained

   sum = 0.0 ;
   for (nfacs=0 ; nfacs<maxfacs ; nfacs++) {
      sum += evals[nfacs] ;     // We have retained this much so far
      if (sum >= frac) {        // Compare it to user's request
         ++nfacs ;              // If there, break out with number of factors
         break ;
         }
      }

/*
   We now do one or two things to avoid expensive operations in the
   'factors' routine.  Equation !!! says that we must divide each factor
   by the square root of the corresponding eigenvalue in order to produce
   unit standard deviations for the factors.
   Simultaneously make most weights positive for user convenience.
   Also, if the user wants standardization of the input variables, divide
   the factor weights by the standard deviations now so that we don't
   have to do it for each case later.
*/

   for (fac=0 ; fac<nfacs ; fac++) {
      if (evals[fac] > 0.0)
         temp = 1.0 / sqrt ( evals[fac] ) ;  // Normalizing factor
      else 
         temp = 0.0 ;
      npos = 0 ;                             // Counts positive weights
      for (var=0 ; var<nvars ; var++) {      // For all vars in this factor
         if (evect[var*nvars+fac] > 0.0)     // If this weight is positive
            ++npos ;                         // Count it
         }
      if (2 * npos < nvars)                  // If fewer than half positive
         temp = -temp ;                      // Flip signs
      for (var=0 ; var<nvars ; var++)
         evect[var*nvars+fac] *= temp ;
      }

   if (std != NULL) {
      for (var=0 ; var<nvars ; var++) {
         for (fac=0 ; fac<nfacs ; fac++)
            evect[var*nvars+fac] /= std[var] ;
         }
      }

   if (std != NULL)
      FREE ( std ) ;
   FREE ( covar ) ;
   FREE ( work2 ) ;  // We keep work1 for use by 'factors'

   *nfactors = nfacs ;
}

/*
   ---> Alternate constructor to read from disk
      Returns (in errnum, conforming to ORTHSAVE.CPP):
         0 - Normal
         2 - Cannot read file
         4 - Insufficient memory
*/

PrincoData::PrincoData ( FILE *fp , int *errnum )
{

   MEMTEXT ( "EIGEN: PrincoData alternate constructor" ) ;

   if (! fread ( &nvars , sizeof(int) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   if (! fread ( &nfacs , sizeof(int) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   means = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evals = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evect = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   work1 = (double *) MALLOC ( nvars * sizeof(double) ) ;

   if ((means == NULL)  ||  (evals == NULL)  ||  (evect == NULL)
    || (work1 == NULL)) {
      if (means != NULL)
         FREE ( means ) ;
      if (evals != NULL)
         FREE ( evals ) ;
      if (evect != NULL)
         FREE ( evect ) ;
      if (work1 != NULL)
         FREE ( work1 ) ;
      nfacs = -1 ;  // Error flag
      *errnum = 4 ;
      return ;
      }

   if (! fread ( means , nvars * sizeof(double) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   if (! fread ( evals , nvars * sizeof(double) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   if (! fread ( evect , nvars * nvars * sizeof(double) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   *errnum = 0 ;
}

/*
   Destructor
*/

PrincoData::~PrincoData ()
{
   if (nfacs < 0)
      return ;
   MEMTEXT ( "EIGEN: PrincoData destructor" ) ;
   FREE ( means ) ;
   FREE ( evals ) ;
   FREE ( evect ) ;
   FREE ( work1 ) ;
}


/*
   factors - Convert raw data to principal factors
*/

void PrincoData::factors (
   int ncases ,      // Number of cases (rows in data array)
   double *data      // Input (ncases by nvars) and output (ncases by nfacs)
   )
{
   int icase, var, fac ;
   double *dptr, sum, diff ;

   for (icase=0 ; icase<ncases ; icase++) { // For each case
      dptr = data + icase * nvars ;         // Point to this case
      for (fac=0 ; fac<nfacs ; fac++) {     // For each factor to be found
         sum = 0.0 ;                        // Will dot evect with case
         for (var=0 ; var<nvars ; var++) {  // All variables in this case
            diff = dptr[var] - means[var] ; // Must be centered
            sum += evect[var*nvars+fac] * diff ; // Cumulate dot product
            }
         work1[fac] = sum ;                 // This case's factor 'fac'
         }
      dptr = data + icase * nfacs ;         // nfacs <= nvars, so may compress
      for (fac=0 ; fac<nfacs ; fac++)       // Replace original variables
         dptr[fac] = work1[fac] ;           // With principal factors
      }
}

/*
   Debugging routine to return pointers to eigenstructure
*/

void PrincoData::eigen ( double **vals , double **vect )
{
   *vals = evals ;
   *vect = evect ;
}

/*
   Save to disk
*/

int PrincoData::save ( FILE *fp )
{
   if (! fwrite ( &nvars , sizeof(int) , 1 , fp ))
      return 1 ;

   if (! fwrite ( &nfacs , sizeof(int) , 1 , fp ))
      return 1 ;

   if (! fwrite ( means , nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   if (! fwrite ( evals , nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   if (! fwrite ( evect , nvars * nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   Discrim - Compute discriminant functions of data array

--------------------------------------------------------------------------------
*/

Discrim::Discrim (
   int ncases ,      // Number of cases (rows in data array)
   int nv ,          // Number of variables (columns in data array)
   int nc ,          // Number of classes
   double *data ,    // Input (ncases by nv) matrix
   int *classes ,    // Class ID of each case (0 to nc-1)
   int maxfacs ,     // Maximum number of output factors
   double frac ,     // Fraction (0-1) of variance to retain
   int *nfactors     // Output: Number of factors retained
   )
{
   int n, icase, iclass, var, var2, fac, *counts, npos ;
   double *dptr, diff1, diff2, sum, temp, *work2 ;
   double *within, *among, *gpmeans, *gmptr, *scratch ;

/*
   Copy parameters to private area and allocate memory.
   Also realize that the number of factors cannot exceed one less than
   the number of classes.
*/

   MEMTEXT ( "EIGEN: Discrim constructor" ) ;
   if (maxfacs > nc-1)
      maxfacs = nc-1 ;

   nvars = nv ;
   nclasses = nc ;

   means = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evals = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evect = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   within = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   among = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   work1 = (double *) MALLOC ( nvars * sizeof(double) ) ;
   work2 = (double *) MALLOC ( nvars * sizeof(double) ) ;
   gpmeans = (double *) MALLOC ( nvars * nclasses * sizeof(double) ) ;
   counts = (int *) MALLOC ( nclasses * sizeof(int) ) ;
   scratch = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;

   if ((means == NULL)  ||  (evals == NULL)  ||  (evect == NULL)  ||
       (within == NULL)  ||  (among == NULL)  ||  (work1 == NULL)  ||
       (work2 == NULL)  ||  (gpmeans == NULL)  ||  (counts == NULL)  ||
       (scratch == NULL)) {
      if (means != NULL)
         FREE ( means ) ;
      if (evals != NULL)
         FREE ( evals ) ;
      if (evect != NULL)
         FREE ( evect ) ;
      if (within != NULL)
         FREE ( within ) ;
      if (among != NULL)
         FREE ( among ) ;
      if (work1 != NULL)
         FREE ( work1 ) ;
      if (work2 != NULL)
         FREE ( work2 ) ;
      if (gpmeans != NULL)
         FREE ( gpmeans ) ;
      if (counts != NULL)
         FREE ( counts ) ;
      if (scratch != NULL)
         FREE ( scratch ) ;
      *nfactors = nfacs = -1 ;  // Error flag
      return ;
      }

/*
   Compute group means and covariances within classes
*/

   memset ( counts , 0 , nclasses * sizeof(int) ) ; // Cases in each class

   n = nvars * nclasses ;           // Zero the group mean vectors
   while (n--)
      gpmeans[n] = 0.0 ;

   n = nvars * nvars ;              // Zero the covariance matrix
   while (n--)
      within[n] = 0.0 ;

   for (icase=0 ; icase<ncases ; icase++) { // Cumulate means
      dptr = data + icase * nvars ;         // Point to this case
      iclass = classes[icase] ;             // Class of this case
      ++counts[iclass] ;                    // Count cases in each class
      gmptr = gpmeans + iclass * nvars ;    // Its mean vector
      for (var=0 ; var<nvars ; var++)       // All variables in this case
         gmptr[var] += dptr[var] ;          // Cumulate mean
      }

   for (iclass=0 ; iclass<nclasses ; iclass++) { // For each class
      if (! counts[iclass])                 // If user is careful, impossible
         counts[iclass] = 1 ;               // But take out cheap insurance
      gmptr = gpmeans + iclass * nvars ;    // Its mean vector
      for (var=0 ; var<nvars ; var++)       // For all variables
         gmptr[var] /= counts[iclass] ;     // Mean is sum divided by n
      }

   for (icase=0 ; icase<ncases ; icase++) { // Cumulate covariances
      dptr = data + icase * nvars ;         // Point to this case
      iclass = classes[icase] ;             // Class of this case
      gmptr = gpmeans + iclass * nvars ;    // Its mean vector
      for (var=0 ; var<nvars ; var++) {
         diff1 = dptr[var] - gmptr[var] ;
         for (var2=var ; var2<nvars ; var2++) {
            diff2 = dptr[var2] - gmptr[var2] ;
            within[var*nvars+var2] += diff1 * diff2 ;
            }
         }
      }

   for (var=0 ; var<nvars ; var++) {         // Divide sums by n
      for (var2=var ; var2<nvars ; var2++) { // To get covariance matrix
         within[var*nvars+var2] /= ncases ;  // This many went into sum
         if (var != var2)                    // Symmetrically duplicate
            within[var2*nvars+var] = within[var*nvars+var2] ;
         }
      }

/*
   Compute the grand mean and the among-classes covariance matrix
*/

   n = nvars ;           // Zero the grand mean vector
   while (n--)
      means[n] = 0.0 ;

   n = nvars * nvars ;   // Zero the covariance matrix
   while (n--)
      among[n] = 0.0 ;

   for (iclass=0 ; iclass<nclasses ; iclass++) {  // For all classes
      gmptr = gpmeans + iclass * nvars ;          // Each group mean vector
      for (var=0 ; var<nvars ; var++)             // gets summed into
         means[var] += gmptr[var] ;               // the grand mean
      }

   for (var=0 ; var<nvars ; var++)                // To get the grand mean
      means[var] /= nclasses ;                    // Must divide by n

   for (iclass=0 ; iclass<nclasses ; iclass++) {  // For all classes
      gmptr = gpmeans + iclass * nvars ;          // Each group mean vector
      for (var=0 ; var<nvars ; var++) {
         diff1 = gmptr[var] - means[var] ;
         for (var2=var ; var2<nvars ; var2++) {
            diff2 = gmptr[var2] - means[var2] ;
            among[var*nvars+var2] += diff1 * diff2 ;
            }
         }
      }

   for (var=0 ; var<nvars ; var++) {         // Divide sums by n
      for (var2=var ; var2<nvars ; var2++) { // To get covariance matrix
         among[var*nvars+var2] /= nclasses ;
         if (var != var2)                    // Symmetrically duplicate
            among[var2*nvars+var] = among[var*nvars+var2] ;
         }
      }

/*
   Compute eigenstructure and number of factors to retain.
   We must copy the within covariance matrix to a scratch area,
   as eigen_bia is destructive, and we need this later.
*/

   for (var=0 ; var<nvars ; var++) {
      for (var2=0 ; var2<nvars ; var2++)
         scratch[var*nvars+var2] = within[var*nvars+var2] ;
      }

   eigen_bia ( nvars , among , scratch , evals , evect , work1 , work2 ) ;

   sum = 0.0 ;
   for (var=0 ; var<nvars ; var++)
      sum += evals[var] ;       // Sum the eigenvalues

   frac *= sum ;                // User wants this much of sum retained

   sum = 0.0 ;
   for (nfacs=0 ; nfacs<maxfacs ; nfacs++) {
      sum += evals[nfacs] ;     // We have retained this much so far
      if (sum >= frac) {        // Compare it to user's request
         ++nfacs ;              // If there, break out with number of factors
         break ;
         }
      }

/*
   Compute the variance of each discriminant function within classes,
   then use this to scale the coefficients to produce unit variance.
   Note that some authorities normalize the total (among+within) variance.
   This is a matter of personal preference and has no practical implications.
   Simultaneously make most weights positive for user convenience.
*/

   for (fac=0 ; fac<nfacs ; fac++) {         // Find variance of each factor
      sum = 0.0 ;                            // Cumulate it here
      for (var=0 ; var<nvars ; var++) {      // It comes from interaction of
         for (var2=0 ; var2<nvars ; var2++)  // all pairs of variables
            sum += evect[var*nvars+fac] * evect[var2*nvars+fac] *
                   within[var*nvars+var2] ;  // Standard quadratic form
         }
      work1[fac] = sum ;  // This factor's variance within classes
      }

   for (fac=0 ; fac<nfacs ; fac++) {         // Normalize each factor
      if (work1[fac] > 0.0)                  // This is the usual case
         temp = 1.0 / sqrt ( work1[fac] ) ;  // Need standard deviation
      else                                   // Pathological case
         temp = 0.0 ;                        // Needs cheap insurance
      npos = 0 ;                             // Counts positive weights
      for (var=0 ; var<nvars ; var++) {      // For all vars in this factor
         if (evect[var*nvars+fac] > 0.0)     // If this weight is positive
            ++npos ;                         // Count it
         }
      if (2 * npos < nvars)                  // If fewer than half positive
         temp = -temp ;                      // Flip signs
      for (var=0 ; var<nvars ; var++)        // For all vars in this factor
         evect[var*nvars+fac] *= temp ;      // Apply factor's scaling factor
      }

   FREE ( within ) ;
   FREE ( among ) ;
   FREE ( gpmeans ) ;
   FREE ( counts ) ;
   FREE ( scratch ) ;
   FREE ( work2 ) ;  // We keep work1 for use by 'factors'

   *nfactors = nfacs ;
}


/*
   ---> Alternate constructor to read from disk
      Returns (in errnum, conforming to ORTHSAVE.CPP):
         0 - Normal
         2 - Cannot read file
         4 - Insufficient memory
*/

Discrim::Discrim ( FILE *fp , int *errnum )
{

   MEMTEXT ( "EIGEN: Discrim alternate constructor" ) ;

   if (! fread ( &nvars , sizeof(int) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   if (! fread ( &nfacs , sizeof(int) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   if (! fread ( &nclasses , sizeof(int) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   means = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evals = (double *) MALLOC ( nvars * sizeof(double) ) ;
   evect = (double *) MALLOC ( nvars * nvars * sizeof(double) ) ;
   work1 = (double *) MALLOC ( nvars * sizeof(double) ) ;

   if ((means == NULL)  ||  (evals == NULL)  ||  (evect == NULL)
    || (work1 == NULL)) {
      if (means != NULL)
         FREE ( means ) ;
      if (evals != NULL)
         FREE ( evals ) ;
      if (evect != NULL)
         FREE ( evect ) ;
      if (work1 != NULL)
         FREE ( work1 ) ;
      nfacs = -1 ;  // Error flag
      *errnum = 2 ;
      return ;
      }

   if (! fread ( means , nvars * sizeof(double) , 1 , fp )) {
      *errnum = 2 ;
      return ;
      }

   if (! fread ( evals , nvars * sizeof(double) , 1 , fp )) {
      *errnum = 1 ;
      return ;
      }

   if (! fread ( evect , nvars * nvars * sizeof(double) , 1 , fp )) {
      *errnum = 1 ;
      return ;
      }

   *errnum = 0 ;
}

/*
   Destructor
*/

Discrim::~Discrim ()
{
   if (nfacs < 0)
      return ;
   MEMTEXT ( "EIGEN: Discrim destructor" ) ;
   FREE ( means ) ;
   FREE ( evals ) ;
   FREE ( evect ) ;
   FREE ( work1 ) ;
}


/*
   factors - Convert raw data to discriminant factors
*/

void Discrim::factors (
   int ncases ,      // Number of cases (rows in data array)
   double *data      // Input (ncases by nvars) and output (ncases by nfacs)
   )
{
   int icase, var, fac ;
   double *dptr, sum, diff ;

   for (icase=0 ; icase<ncases ; icase++) { // For each case
      dptr = data + icase * nvars ;         // Point to this case
      for (fac=0 ; fac<nfacs ; fac++) {     // For each factor to be found
         sum = 0.0 ;                        // Will dot evect with case
         for (var=0 ; var<nvars ; var++) {  // All variables in this case
            diff = dptr[var] - means[var] ; // Must be centered
            sum += evect[var*nvars+fac] * diff ; // Cumulate dot product
            }
         work1[fac] = sum ;                 // This case's factor 'fac'
         }
      dptr = data + icase * nfacs ;         // nfacs <= nvars, so may compress
      for (fac=0 ; fac<nfacs ; fac++)       // Replace original variables
         dptr[fac] = work1[fac] ;           // With principal factors
      }
}

/*
   Debugging routine to return pointers to eigenstructure
*/

void Discrim::eigen ( double **vals , double **vect )
{
   *vals = evals ;
   *vect = evect ;
}

/*
   Save to disk
*/

int Discrim::save ( FILE *fp )
{
   if (! fwrite ( &nvars , sizeof(int) , 1 , fp ))
      return 1 ;

   if (! fwrite ( &nfacs , sizeof(int) , 1 , fp ))
      return 1 ;

   if (! fwrite ( &nclasses , sizeof(int) , 1 , fp ))
      return 1 ;

   if (! fwrite ( means , nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   if (! fwrite ( evals , nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   if (! fwrite ( evect , nvars * nvars * sizeof(double) , 1 , fp ))
      return 1 ;

   return 0 ;
}
