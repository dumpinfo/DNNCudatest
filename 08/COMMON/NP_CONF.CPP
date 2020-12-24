/******************************************************************************/
/*                                                                            */
/*  NP_CONF - NPconf object routines for nonparametric confidence intervals   */
/*    This also includes several routines for binomial confidences that are   */
/*    not part of the NPconf object.                                          */
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

#define DEBUG 0

/*
--------------------------------------------------------------------------------

   Constructor, destructor

   The private 'intervals' is a little tricky.  This is an array of nvars
   pointers to npred pairs of confidence limits.  The 'conf' member function
   fills them in and passes them to the corresponding signal objects.
   It then sets 'intervals' to NULL as a flag to the NPconf destructor
   to avoid freeing them.  They will be freed by the Signal destructor.
   In the event that 'conf' never gets called, the NPconf destructor frees
   them.
   
--------------------------------------------------------------------------------
*/

NPconf::NPconf (
   int nvars ,            // Number of variables
   int npred ,            // Number of prediction distances
   int ncases ,           // Number of cases at each distance
   double ttprob ,        // Two-tailed probability (typically .9)
   int n_conf_comps ,     // Number of confidence compensations
   ConfComp *conf_comps   // This is them
   )
{
   int i, j ;

   nv = nvars ;
   np = npred ;
   nc = ncases ;
   ncc = n_conf_comps ;
   ccs = conf_comps ;
   prob = ttprob ;

   ok = 0 ;
   MEMTEXT ( "NP_CONF: constructor" ) ;
   x = (double *) MALLOC ( nvars * npred * ncases * sizeof(double) ) ;
   known = (double *) MALLOC ( nvars * npred * sizeof(double) ) ;
   pred = (double *) MALLOC ( nvars * npred * sizeof(double) ) ;
   isigs = (int *) MALLOC ( nvars * sizeof(int) ) ;
   sigs = (Signal **) MALLOC ( nvars * sizeof(Signal *) ) ;
   intervals = (double **) MALLOC ( nvars * sizeof(double *) ) ;
   ccwork = (double **) MALLOC ( nvars * sizeof(double *) ) ;
   if ((x == NULL) || (known == NULL) || (pred == NULL) || (isigs == NULL)
    || (sigs == NULL)  || (intervals == NULL)  || (ccwork == NULL)) {
      if (x != NULL)
         FREE ( x ) ;
      if (known != NULL)
         FREE ( known ) ;
      if (pred != NULL)
         FREE ( pred ) ;
      if (isigs != NULL)
         FREE ( isigs ) ;
      if (sigs != NULL)
         FREE ( sigs ) ;
      if (intervals != NULL)
         FREE ( intervals ) ;
      if (ccwork != NULL)
         FREE ( ccwork ) ;
      x = known = pred = NULL ;
      isigs = NULL ;
      sigs = NULL ;
      intervals = ccwork = NULL ;
      return ;
      }

   for (i=0 ; i<nvars ; i++) {
      ccwork[i] = NULL ;
      intervals[i] = (double *) MALLOC ( 2 * npred * sizeof(double) ) ;
      if (intervals[i] == NULL) {
         for (j=0 ; j<i ; j++)
            FREE ( intervals[j] ) ;
         FREE ( intervals ) ;
         FREE ( x ) ;
         FREE ( known ) ;
         FREE ( pred ) ;
         FREE ( isigs ) ;
         FREE ( sigs ) ;
         x = known = pred = NULL ;
         isigs = NULL ;
         sigs = NULL ;
         intervals = ccwork = NULL ;
         }
      }

   ok = 1 ;
}

NPconf::~NPconf ()
{
   int i ;

   if (! ok)
      return ;
   MEMTEXT ( "NP_CONF: destructor" ) ;
   if (x != NULL)
      FREE ( x ) ;
   if (known != NULL)
      FREE ( known ) ;
   if (pred != NULL)
      FREE ( pred ) ;
   if (isigs != NULL)
      FREE ( isigs ) ;
   if (sigs != NULL)
      FREE ( sigs ) ;
   if (intervals != NULL) {
      for (i=0 ; i<nv ; i++)
         FREE ( intervals[i] ) ;
      FREE ( intervals ) ;
      }
   if (ccwork != NULL) {
      for (i=0 ; i<nv ; i++) {
         if (ccwork[i] != NULL)
            FREE ( ccwork[i] ) ;
         }
      FREE ( ccwork ) ;
      }
}

/*
--------------------------------------------------------------------------------

   sig_init - Initialize signal pointers

--------------------------------------------------------------------------------
*/

int NPconf::sig_init (
   int ivar ,       // Ordinal variable number
   int isig ,       // Corresponds to this signal
   Signal *sig      // This is its pointer
   )
{
   int i, n, icc, p, pass, nfound, passes, period, neach ;
   double *dptr, *prevdata, sum, temp, xmean, rate ;
   ConfComp *ccptr ;
   Signal *sptr ;

/*
   Save this signal's ordinal number and its pointer
*/

   isigs[ivar] = isig ;
   sigs[ivar] = sig ;

/*
   Normally, confidence intervals are additive.  If the first transform is
   a log, we make them multiplicative.
*/

   sig->mult_conf = 0 ;   // Additive is the usual case.

/*
   Any confidence compensations for it?
*/

   n = nfound = 0 ;                   // Counts need beyond signal length
   for (i=0 ; i<ncc ; i++) {          // Check all confidence compensations
      ccptr = ccs+i ;                 // Point to this one
      if (ccptr->signum != isig)      // If not this one
         continue ;                   // Keep going through them
      ccptr->ord_num = nfound ;       // Ordinal number (org 0) for initial log
      switch (ccptr->which) {
         case CCcenter:
            ++nfound ;
            break ;
         case CCdetrend:
            ++nfound ;
            break ;
         case CCoffset:
            ++nfound ;
            break ;
         case CCscale:
            ++nfound ;
            break ;
         case CCstandardize:
            ++nfound ;
            break ;
         case CCdifference:
            n += ccptr->persig->differenced ;
            nfound += ccptr->persig->differenced ; // One for each difference
            break ;
         case CCseasonal:
            n += ccptr->persig->seasonal ;
            ++nfound ;
         case CClog:
            if (! nfound)
               sig->mult_conf = 1 ;
            ++nfound ;
            break ;
         }
      }

   if (! nfound) {                     // If no confidence compensations, done
      ccwork[ivar] = NULL ;
      return 0 ;
      }

/*
   Allocate memory for the unmodified signal, then compute it
*/

   neach = sig->n + n ;   // This many cases in each undoing
   MEMTEXT ( "NP_CONF: sig_init ccwork" ) ;
   dptr = ccwork[ivar] = (double *) MALLOC ( nfound * neach * sizeof(double) ) ;
   if (dptr == NULL)
      return 1 ;

   n = sig->n ;
   prevdata = sig->sig ;

   for (icc=ncc-1 ; icc>=0 ; icc--) { // Check all confidence compensations
      ccptr = ccs+icc ;               // Point to this one
      if (ccptr->signum != isig)      // If not this one
         continue ;                   // Keep going through them
      sptr = ccptr->persig ;          // This signal has the parameters
      
      switch (ccptr->which) {

         case CCcenter:
            if (! sptr->centered)
               break ;
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            for (i=0 ; i<n ; i++)
               dptr[i] += sptr->cent ;
            prevdata = dptr ;
            dptr += n ;
            break ;

         case CCdetrend:
            if (! sptr->detrended)
               break ;
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            rate = sptr->slope ;
            xmean = sptr->pivot ;
            for (i=0 ; i<n ; i++)
               dptr[i] += (i - xmean) * rate ;
            prevdata = dptr ;
            dptr += n ;
            break ;

         case CCoffset:
            if (! sptr->offsetted)
               break ;
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            for (i=0 ; i<n ; i++)
               dptr[i] -= sptr->offst ;
            prevdata = dptr ;
            dptr += n ;
            break ;

         case CCscale:
            if ((! sptr->scaled)  ||  (sptr->scal == 0.0))
               break ;
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            for (i=0 ; i<n ; i++)
               dptr[i] /= sptr->scal ;
            prevdata = dptr ;
            dptr += n ;
            break ;

         case CCstandardize:
            if (! sptr->standardized)
               break ;
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            for (i=0 ; i<n ; i++)
               dptr[i] = sptr->std * dptr[i] + sptr->mean ;
            prevdata = dptr ;
            dptr += n ;
            break ;

         case CCdifference:
            passes = sptr->differenced ;
            if (! passes)
               break ;
            for (pass=0 ; pass<passes ; pass++) {
               memcpy ( dptr , prevdata , n * sizeof(double) ) ;
               sum = sptr->leads[passes-pass-1] ;
               for (i=0 ; i<n ; i++) {
                  temp = dptr[i] ;
                  dptr[i] = sum ;
                  sum += temp ;
                  }
               dptr[i] = sum ;
               ++n ;
               prevdata = dptr ;
               dptr += n ;
               }
            break ;

         case CCseasonal:
            period = sptr->seasonal ;
            if (! period)
               break ;
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            for (p=0 ; p<period ; p++) {
               sum = sptr->sleads[p] ;
               for (i=p ; i<n ; i+=period ) {
                  temp = dptr[i] ;
                  dptr[i] = sum ;
                  sum += temp ;
                  }
               dptr[i] = sum ;
               }
            n += period ;
            prevdata = dptr ;
            dptr += n ;
            break ;

         case CClog:
            memcpy ( dptr , prevdata , n * sizeof(double) ) ;
            if (ccptr->ord_num) {       // Do not do log if first transform
               for (i=0 ; i<n ; i++) {
                  if (dptr[i] > 200.0)
                     dptr[i] = 1.e87 ;
                  else 
                     dptr[i] = exp ( dptr[i] ) ;
                  }
               }
            prevdata = dptr ;
            dptr += n ;
            break ;
         }
      }

   return 0 ;
}


/*
--------------------------------------------------------------------------------

   insert - Insert a pair of known and predicted values into temporary area

--------------------------------------------------------------------------------
*/

void NPconf::insert (
   int ivar ,       // Ordinal variable number
   int idist ,      // Prediction distance
   double xknown ,  // True value
   double xpred     // Predicted value
   )
{
   known[ivar*np+idist] = xknown ;
   pred[ivar*np+idist] = xpred ;
}

/*
--------------------------------------------------------------------------------

   eval - Evaluate the error for a set of predictions (after inserts)

--------------------------------------------------------------------------------
*/

void NPconf::eval ( int casenum , int offset )
{
   int ivar, ipred, icc, this_sig, ipass, period, n ;
   int npasses ;
   double *kptr, *pptr, *dptr, xmean, rate ;
   ConfComp *ccptr ;
   Signal *sigptr ;

   for (ivar=0 ; ivar<nv ; ivar++) {
      this_sig = isigs[ivar] ;           // Which ordinal signal is this output?
      sigptr = sigs[ivar] ;              // Its pointer
      pptr = pred + ivar * np ;

      n = 0 ;                            // Offset due to undoings
      dptr = ccwork[ivar] ;              // Sig_init put undoings here
      kptr = sigptr->sig ;               // This will point to known values
      for (icc=ncc-1 ; icc>=0 ; icc--) { // Process all confidence compensations
         ccptr = ccs+icc ;               // Point to this one
         if (ccptr->signum != this_sig)  // If not this one
            continue ;                   // Keep going through them

         switch (ccptr->which) {

            case CCcenter:
               if (! ccptr->persig->centered)
                  break ;
               for (ipred=0 ; ipred<np ; ipred++)
                  pptr[ipred] += ccptr->persig->cent ;
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;

            case CCdetrend:
               if (! ccptr->persig->detrended)
                  break ;
               rate = ccptr->persig->slope ;
               xmean = ccptr->persig->pivot ;
               for (ipred=0 ; ipred<np ; ipred++)
                  pptr[ipred] += (casenum + ipred - xmean) * rate ;
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;

            case CCoffset:
               if (! ccptr->persig->offsetted)
                  break ;
               for (ipred=0 ; ipred<np ; ipred++)
                  pptr[ipred] -= ccptr->persig->offst ;
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;

            case CCscale:
               if ((! ccptr->persig->scaled)  ||  (ccptr->persig->scal == 0.0))
                  break ;
               for (ipred=0 ; ipred<np ; ipred++)
                  pptr[ipred] /= ccptr->persig->scal ;
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;

            case CCstandardize:
               if (! ccptr->persig->standardized)
                  break ;
               for (ipred=0 ; ipred<np ; ipred++)
                  pptr[ipred] = ccptr->persig->std * pptr[ipred] +
                                ccptr->persig->mean ;
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;

            case CCdifference:
               if (! ccptr->persig->differenced)
                  break ;
               npasses = ccptr->persig->differenced ;
               for (ipass=0 ; ipass<npasses ; ipass++) {
                  ++n ;
                  pptr[0] += dptr[casenum+n-1] ;
                  for (ipred=1 ; ipred<np ; ipred++)
                     pptr[ipred] += pptr[ipred-1] ;
                  kptr = dptr ;            // Keep track of known values
                  dptr += sigptr->n + n ;  // Point to next undoing record
                  }
               break ;

            case CCseasonal:
               if (! ccptr->persig->seasonal)
                  break ;
               period = ccptr->persig->seasonal ;
               n += period ;
               for (ipred=0 ; (ipred<period) && (ipred<np) ; ipred++)
                  pptr[ipred] += dptr[casenum+n+ipred-period] ;
               if (period > 0) {   // Insurance
                  for (ipred=period ; ipred<np ; ipred++)
                     pptr[ipred] += pptr[ipred-period] ;
                  }
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;

            case CClog:
               if (ccptr->ord_num) {    // Do not do log if first transform
                  for (ipred=0 ; ipred<np ; ipred++) {
                     if (pptr[ipred] > 200.0)
                        pptr[ipred] = 1.e87 ;
                     else 
                        pptr[ipred] = exp ( pptr[ipred] ) ;
                     }
                  }
               kptr = dptr ;            // Keep track of known values
               dptr += sigptr->n + n ;  // Point to next undoing record
               break ;
            } // Switch on operations
         } // For all confidence compensations

#if DEBUG > 2
      printf ( "\nIVAR=%d", ivar ) ;
      for (ipred=0 ; ipred<np ; ipred++)
         printf ( " (%.2lf %.2lf)", kptr[casenum+n+ipred], pptr[ipred] ) ;
#endif
      for (ipred=0 ; ipred<np ; ipred++)
         x [ nc * (np * ivar + ipred) + casenum - offset ] =
            kptr[casenum+n+ipred] - pptr[ipred] ;
      }
}

/*
--------------------------------------------------------------------------------

   conf - Compute the final confidence intervals

   This returns in excess_tail_prob the probability associated with the
   chosen limit giving a shaded (error) area of 0.05.
   In other words, there is a 5-percent chance that the lower confidence
   limit is really excess_tail_prob, not the user's request.  The same
   goes for the upper limit.

   This returns in cover_prob the probability that the computed confidence
   limits have tails that are TWICE the requested area or better.  In other
   words, suppose the user asks for a 90-percent interval.  Cover_prob is
   the probability that it is at least an 80-percent interval.



--------------------------------------------------------------------------------
*/

void NPconf::conf ( double *excess_tail_prob , double *cover_prob )
{
   int ivar, ipred, lower, upper ;
   double *dptr, *cptr, p, pp, std ;
   Signal *sptr ;

   *excess_tail_prob = *cover_prob = -1.0 ;

   if (intervals == NULL)    // This should NEVER happen
      return ;               // So this is just cheap insurance

   lower = 0.5 * (1.0 - prob) * nc - 1 ;
   if (lower < 0)
      lower = 0 ;
   upper = nc - lower - 1 ;
#if DEBUG
   printf ( "\nNC=%d  lower=%d  upper=%d", nc, lower, upper ) ;
#endif

   for (ivar=0 ; ivar<nv ; ivar++) { // Do each output signal
      cptr = intervals[ivar] ;       // Our constructor allocd this array
      sptr = sigs[ivar] ;            // Which is for this signal
      sptr->npred = np ;             // Tell the signal how many predictions
      if (sptr->intervals != NULL)   // There may be an old one there
         FREE ( sptr->intervals ) ;
      sptr->intervals = cptr ;       // Conf limits (Freed by Signal destructor)
      for (ipred=0 ; ipred<np ; ipred++) {
         dptr = x + nc * (np * ivar + ipred) ;
         qsort ( 0 , nc-1 , dptr ) ;
         if (sptr->mult_conf) {
            cptr[2*ipred] = exp ( dptr[lower] ) ;
            cptr[2*ipred+1] = exp ( dptr[upper] ) ;
            }
         else {
            cptr[2*ipred] = dptr[lower] ;
            cptr[2*ipred+1] = dptr[upper] ;
            }
#if DEBUG
         printf ( "\n  ivar=%d  ipred=%2d   (%10.4lf %10.4lf)",
                  ivar, ipred, dptr[lower], dptr[upper] ) ;
#endif
         }
      }
#if DEBUG > 1
   getch () ;
#endif

   FREE ( intervals ) ;
   intervals = NULL ;  // Flag NPconf destructor to avoid freeing this
                       // The Signal destructor will free the arrays

   if (nc > 1000) {    // Use a normal approximation
      p = (lower + 1.5) / (double) nc ;
      std = sqrt ( nc * p * (1.0 - p) ) / nc ;
      for (ivar=0 ; ivar<12 ; ivar++) { // Fixed-point iteration improves it
         pp = p + 1.64 * std ;
         std = sqrt ( nc * pp * (1.0 - pp) ) / nc ;
         }
      p += 1.64 * std ;
      *excess_tail_prob = p ;
      *cover_prob = 1.0 ;
      }
   else {
      *excess_tail_prob = p_limit ( nc , lower+1 , 0.05 ) ;
      *cover_prob = t_limit ( nc , lower+1 , 1.0 - 2.0 * (1.0 - prob) ) ;
      }
}

/*
--------------------------------------------------------------------------------

   These routines are not for the NPconf class.
   They are for binomial confidences.

--------------------------------------------------------------------------------
*/

#define P_LIMIT_INC 0.1
#define P_LIMIT_ACC 1.e-7

double binom ( int n , int m )
{
   double x = 1.0 ;

   while (m) {
      x *= n-- ;
      x /= m-- ;
      }

   return x ;
}

double binom_sum ( int start , int stop , int n , double p )
{
   int i ;
   double sum, log_p, log_1mp ;

   if (p <= 0.0)
      p = 1.e-14 ;

   if (p >= 1.0)
      p = 1.0 - 1.e-14 ;

   log_p = log ( p ) ;
   log_1mp = log ( 1.0 - p ) ;

   sum = 0.0 ;
   for (i=start ; i<=stop ; i++)
      sum += binom ( n , i ) * exp ( i * log_p ) * exp ( (n-i) * log_1mp ) ;

   return sum ;
}

double p_limit ( int n , int m , double limit )
{
   int i ;
   double p1, p2, p3, y1, y2, y3, denom, trial, y ;

/*
   The first point is known
*/

   p1 = 0.0 ;
   y1 = limit - 1.0 ;

/*
   This scans upward until it changes sign
*/

   for (p3=P_LIMIT_INC ; p3<1.0 ; p3+=P_LIMIT_INC) {
      y3 = limit - binom_sum ( 0 , m , n , p3 ) ;
      if (fabs ( y3 ) < P_LIMIT_ACC)   // Very unlikey
         return p3 ;   // But check anyway
      if (y3 > 0.0)    // When we change sign
         break ;       // The root is bracketed
      p1 = p3 ;        // Move on to next trial interval
      y1 = y3 ;
      }

/*
   The root has been bracketed between (p1, y1) and (p3, y3).  Refine.
*/

   for (i=0 ; i<100 ; i++) {       // This limit is insurance that is never used
      p2 = 0.5 * (p1 + p3) ;       // Midpoint of current bounding interval
      if (p3 - p1  <  P_LIMIT_ACC) // This is a secondary convergence test
         return p2 ;
      y2 = limit - binom_sum ( 0 , m , n , p2 ) ;  // Compute criterion
      if (fabs ( y2 ) < P_LIMIT_ACC)  // This is a main convergence test
         return p2 ;
      denom = sqrt ( y2 * y2  -  y1 * y3 ) ; // Remember y1, y3 opposite sign
      trial = p2 + (p1 - p2) * y2 / denom ;  // New test point for root
      y = limit - binom_sum ( 0 , m , n , trial ) ; // Compute criterion
      if (fabs ( y ) < P_LIMIT_ACC)   // This is another main convergence test
         return trial ;
      if ((y2 < 0.0)  &&  (y > 0.0)) { // Root between midpoint and test point?
         p1 = p2 ;
         y1 = y2 ;
         p3 = trial ;
         y3 = y ;
         }
      else if ((y < 0.0)  &&  (y2 > 0.0)) { // Ditto, but other way?
         p1 = trial ;
         y1 = y ;
         p3 = p2 ;
         y3 = y2 ;
         }
      else if (y < 0.0) {  // Must keep one of the current bounds
         p1 = trial ;
         y1 = y ;
         }
      else {               // Ditto, but other bound
         p3 = trial ;
         y3 = y ;
         }
      }

   return -1.0 ; // Only get here if convergence failure  
}

double t_limit ( int n , int m , double limit )
{
   return 1.0 - binom_sum ( n-2*m+1 , n , n , limit ) ;
}
