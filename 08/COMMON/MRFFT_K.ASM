; Copyright (c) 1995 Timothy Masters.  All rights reserved.
; Reproduction or translation of this work beyond that permitted in section
; 117 of the 1976 United States Copyright Act without the express written
; permission of the copyright owner is unlawful.  Requests for further
; information should be addressed to the Permissions Department, John Wiley
; & Sons, Inc.  The purchaser may make backup copies for his/her own use
; only and not for distribution or resale.
; Neither the author nor the publisher assumes responsibility for errors,
; omissions, or damages, caused by the use of these programs or from the
; use of the information contained herein.

.586c
.587
_TEXT	segment dword use32 public 'CODE'
_TEXT	ends
_DATA	segment dword use32 public 'DATA'
_DATA	ends
_BSS	segment dword use32 public 'BSS'
_BSS	ends
DGROUP	group	_DATA,_BSS
	extrn	_sin
	extrn	_cos
	extrn	_abs

	public	_kernels
_TEXT	segment
	assume	CS:_TEXT
;
;void kernels ( double *real , double *imag , int ntot , int npts , int nspan ,
;               int isign , int n_facs , double *rwork , double *iwork ,
;               double *cosines , double *sines , int *factors )
;{
_kernels:
		enter	0184h,0
		push	EBX
		push	ESI
		push	EDI
;   int j, k, l, m, ibase, inner_span, full_span, facnum, itrig ;
;   int last_point, previous_kernel, limit, kernel, inc, jump, tot_pts ;
;   double c0, s0, c1, s1, c2, s2, c3, s3, r0, i0 ;
;   double sin_third, sin_fifth, cos_fifth ;
;   double angle, rtemp, itemp, rtemp2, itemp2, temp ;
;   double real_sum, real_diff, real_sum2, real_diff2 ;
;   double imag_sum, imag_diff, imag_sum2, imag_diff2 ;
;   double *rptr0, *iptr0, *rptr1, *iptr1, *rptr2, *iptr2, *rptr3, *iptr3 ;
;   double *rptr4, *iptr4, *flagptr, *flagptr2, *endptr, *finalptr ;
;
;   inc = abs ( isign ) ;
		push	dword ptr 01Ch[EBP]
		call	near ptr _abs
		add	ESP,4
		mov	-0154h[EBP],EAX
;   tot_pts = inc * ntot ;
		imul	EAX,010h[EBP]
		mov	-014Ch[EBP],EAX
;   last_point = tot_pts - inc ;
		sub	EAX,-0154h[EBP]
		mov	-0160h[EBP],EAX
;   inner_span = inc * nspan ;
		mov	EAX,-0154h[EBP]
		imul	EAX,018h[EBP]
		mov	-0170h[EBP],EAX
;   jump = inner_span / npts ;
		cdq
		idiv	dword ptr 014h[EBP]
		mov	-0150h[EBP],EAX
;
;/*
;   Compute some trig values that may be used later.
;   These are for a third and a fifth of a circle,
;   for the kernels 3 and 5, respectively.
;*/
;
;   sin_third = sin ( 2.0 * PI / 3.0 ) ;
                fldpi
                fld1
                fadd    st,st(0)
                fld1
                fxch
                fadd    st(1),st
                fmulp   st(2),st
                fdivp   st(1),st
                fsin
		fstp	qword ptr -0F8h[EBP]
;   sin_fifth = sin ( 2.0 * PI / 5.0 ) ;
;   cos_fifth = cos ( 2.0 * PI / 5.0 ) ;
                fldpi
                fld1
                fadd    st,st(0)
                fld1
                fxch
                fadd    st(1),st
                fadd    st(1),st
                fmulp   st(2),st
                fdivp   st(1),st
                fsincos
		fstp	qword ptr -0E8h[EBP]
		fstp	qword ptr -0F0h[EBP]
;
;   if (isign < 0) {
		cmp	dword ptr 01Ch[EBP],0
		jns	LC6
;      sin_third = -sin_third ;
		fld	qword ptr -0F8h[EBP]
		fchs
		fstp	qword ptr -0F8h[EBP]
;      sin_fifth = -sin_fifth ;
		fld	qword ptr -0F0h[EBP]
		fchs
		fstp	qword ptr -0F0h[EBP]
;      }
;
;/*
;   Main Fourier loop
;*/
;
;   facnum = 0 ;            // Indexes the factors
LC6:		xor	EAX,EAX
		mov	-0168h[EBP],EAX
;   previous_kernel = 0 ;   // Only recompute trig if kernel changes
		mov	-015Ch[EBP],EAX
;
;kernel_loop:
;   angle = 2.0 * PI * jump / (double) inner_span ;
LD4:		fld	dword ptr -0150h[EBP]
                fldpi
                fadd    st,st
		fmulp   st(1),st
		fdiv	dword ptr -0170h[EBP]
		fst	qword ptr -0E0h[EBP]
;   if (isign < 0)
		cmp	dword ptr 01Ch[EBP],0
		jns	L102
;      angle = -angle ;
		fchs
		fst	qword ptr -0E0h[EBP]
;   temp = sin ( 0.5 * angle ) ;
L102:
		fld1
                fadd    st,st
                fdivp   st(1),st
                fsin
		fstp	qword ptr -0B8h[EBP]
;   c0 = 2.0 * temp * temp ;
		fld	qword ptr -0B8h[EBP]
		fmul	qword ptr -0B8h[EBP]
                fadd    st,st
		fstp	qword ptr -0148h[EBP]
;   s0 = sin ( angle ) ;
		fld	qword ptr -0E0h[EBP]
                fsin
		fstp	qword ptr -0140h[EBP]
;   ibase = 0 ;
		mov	dword ptr -0174h[EBP],0
;   kernel = factors[facnum++] ;
		mov	EAX,-0168h[EBP]
		inc	dword ptr -0168h[EBP]
		mov	ECX,034h[EBP]
		mov	EAX,[EAX*4][ECX]
		mov	-0158h[EBP],EAX
;
;/*
;   Kernel of 2
;*/
;
;   if (kernel == 2) {
		cmp	EAX,2
		jne	L49D
;      inner_span /= 2 ;
                shr     dword ptr -0170h[ebp],1
;
;      rptr0 = real ;
;      iptr0 = imag ;
                mov     edi,8[ebp]
                mov     esi,0Ch[ebp]
;      flagptr = real + last_point ;
		mov	ECX,-0160h[EBP]
		lea	ECX,[ECX*8][edi]
		mov	-04Ch[EBP],ECX
;      endptr = real + jump ;
		mov	ECX,-0150h[EBP]
		lea	EAX,[ECX*8][edi]
		mov	-044h[EBP],EAX

		mov	ebx,-0170h[EBP]    ; inner_span kept here throughout
;      for (;;) {
;         rptr1 = rptr0 + inner_span ;
L1C5:
		lea	eax,[ebx*8][edi]   ; Keep rptr1 in eax (ebx=inner_span)
;         temp = *rptr1 ;
;         *rptr1 = *rptr0 - temp ;
;         *rptr0 += temp ;
		fld	qword ptr [EAX]
		fld	qword ptr [edi]
		fsub	st,st(1)
		fstp	qword ptr [EAX]
		fadd	qword ptr [edi]
                fstp    qword ptr [edi]
;         iptr1 = iptr0 + inner_span ;
		lea	ecx,[ebx*8][esi]   ; Keep iptr1 in ecx
;         temp = *iptr1 ;
;         *iptr1 = *iptr0 - temp ;
;         *iptr0 += temp ;
		fld	qword ptr [ECX]
		fld	qword ptr [esi]
		fsub	st,st(1)
		fstp	qword ptr [ECX]
		fadd	qword ptr [esi]
		fstp	qword ptr [esi]

;         rptr0 = rptr1 + inner_span ;
;         iptr0 = iptr1 + inner_span ;
		lea	edi,[ebx*8][eax]  ; eax is rptr1
		lea	esi,[ebx*8][ecx]  ; ecx is iptr1
;         if (rptr0 < flagptr)
		cmp	edi,-04Ch[EBP]
		jb	L1C5    ; Continue

;         rptr0 -= last_point ;
;         iptr0 -= last_point ;
		mov	EAX,-0160h[EBP]
		shl	EAX,3
                sub     edi,eax
                sub     esi,eax
;         if (rptr0 >= endptr)
;            break ;
		cmp	edi,-044h[EBP]
		jb	L1C5

;      if (rptr0 - real >= inner_span)
;         return ;
                mov     eax,edi
		sub	eax,8[EBP]
		shr     eax,3
		cmp	eax,-0170h[EBP]
		jl	L271
		pop	EDI
		pop	ESI
		pop	EBX
		leave
		ret
;
;      k = inner_span + 2 ;
L271:		mov	EAX,-0170h[EBP]
		add	EAX,2
		mov	-0180h[EBP],EAX

;  Start of second outer loop in kernel 2
;  We push c0 and s0 to keep them available in the stack
;  At the start of each pass through this loop, also push c1 and s1.
;  Pop them at the bottom of the loop to keep it clean.
;  Register usage:
;    edi = rptr0
;    esi = iptr0
;    ebx = inner_span
;    eax = rptr1
;    ecx = iptr1

		fld	qword ptr -0148h[EBP]  ; c0
		fld	qword ptr -0140h[EBP]  ; s0

		mov	ebx,-0170h[EBP]        ; inner_span

;      for (;;) {
;         c1 = 1.0 - c0 ;
L280:		fld1
		fsub    st,st(2)
;         s1 = s0 ;
		fld	st(1)
;
;         flagptr = real + tot_pts - 1 ;
		mov	ECX,-014Ch[EBP]
		mov	EDX,8[EBP]
                dec     ecx
		lea	ECX,[ECX*8][EDX]
		mov	-04Ch[EBP],ECX
;         limit = k / 2 ;
;         endptr = real + limit - jump ;
		mov	EAX,-0180h[EBP]          ; k
		sar	EAX,1
		sub	eax,-0150h[EBP]          ; jump
		lea	EAX,[EAX*8][EDX]
		mov	-044h[EBP],EAX
;         for (;;) {
;            rptr1 = rptr0 + inner_span ;
L2CA:           lea     eax,[ebx*8][edi]         ; ebx=inner_span, edi=rptr0
;            rtemp = *rptr0 - *rptr1 ;
		fld	qword ptr [edi]
		fsub	qword ptr [eax]
		fstp	qword ptr -0D8h[EBP]
;            *rptr0 += *rptr1 ;
		fld	qword ptr [eax]
		fadd	qword ptr [edi]
		fstp	qword ptr [edi]
;            iptr1 = iptr0 + inner_span ;
                lea     ecx,[ebx*8][esi]         ; ebx=inner_span, esi=iptr0
;            itemp = *iptr0 - *iptr1 ;
		fld	qword ptr [esi]
		fsub	qword ptr [ecx]
		fstp	qword ptr -0D0h[EBP]
;            *iptr0 += *iptr1 ;
		fld	qword ptr [ecx]
		fadd	qword ptr [esi]
		fstp	qword ptr [esi]

;            *rptr1 = c1 * rtemp - s1 * itemp ;
		fld	qword ptr -0D8h[EBP]
		fmul    st,st(2)                  ; c1
		fld	qword ptr -0D0h[EBP]
                fmul    st,st(2)                  ; s1
		fsubp	ST(1),ST
		fstp	qword ptr [EAX]
;            *iptr1 = s1 * rtemp + c1 * itemp ;
		fld	qword ptr -0D8h[EBP]
		fmul    st,st(1)                  ; s1
		fld	qword ptr -0D0h[EBP]
		fmul    st,st(3)                  ; c1
		faddp	ST(1),ST
		fstp	qword ptr [ECX]
;            rptr0 += 2 * inner_span ;
;            iptr0 += 2 * inner_span ;
		mov	edx,ebx
		shl	edx,4
                add     edi,edx
                add     esi,edx
;            if (rptr0 < flagptr)
;               continue
		cmp	edi,-04Ch[EBP]
		jb	near ptr L2CA   ; Continue

;            j = rptr0 - real ;
;            j = k + tot_pts - 2 * j - 2 ;
                mov     eax,edi                  ; rptr0
		mov	ECX,-0180h[EBP]          ; k
		sub	EAX,8[EBP]               ; real
		add	ECX,-014Ch[EBP]          ; tot_pts
                shr     eax,2
		sub	ECX,2
		sub	ECX,EAX
		mov	-0184h[EBP],ECX
;            rptr0 += j ;
		shl	ECX,3
                add     edi,ecx
;            iptr0 += j ;
                add     esi,ecx
;            c1 = -c1 ;
		fld	st(1)
		fchs
                fstp     st(2)
;            if (tot_pts + j  >  0)
		mov	EAX,-014Ch[EBP]
		add	EAX,-0184h[EBP]
		jg	near ptr L2CA
;            if (rptr0 >= endptr)
		cmp	edi,-044h[EBP]
		jae short	L440
;            rptr0 += jump ;
;            iptr0 += jump ;
		mov	EAX,-0150h[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax
;            temp = c1 - (c0 * c1 + s0 * s1) ;
		fld	st(3)                    ; c0
		fmul	st,st(2)                 ; c1
		fld	st(3)                    ; s0
		fmul	st,st(2)                 ; s1
		faddp	ST(1),ST                 ; Sum now in st(0)
;            s1 = (s0 * c1 - c0 * s1) + s1 ;
		fld	st(3)                    ; s0
		fmul	st,st(3)                 ; c1
		fld	st(5)                    ; c0
		fmul	st,st(3)                 ; s1
		fsubp	ST(1),ST                 ; Diff now in st(0)
		faddp	st(2),st                 ; New s1 now in st(1)
;            c1 = temp ;
                fsubp   st(2),st                 ; st(0) held c0 * c1 + s0 * s1
		jmp	near ptr L2CA
;            }
;
;         k += inc * 2 ;
L440:		mov	EAX,-0154h[EBP]
		shl	EAX,1
		add	-0180h[EBP],EAX
;         ibase = (k - inner_span) / 2 + jump - 1 ;
		mov	EAX,-0180h[EBP]
		sub	EAX,ebx
		sar	EAX,1
		add	EAX,-0150h[EBP]
		dec	EAX
;         if (ibase >= 2 * jump)
		mov	ECX,-0150h[EBP]
		shl	ECX,1
		cmp	ECX,EAX
		jg	L47D
;  Done with second loop of Kernel 2.  Clean stack, jump to top of kernel loop.
                finit
		jmp	near ptr LD4

;        rptr0 = real + ibase ;
;        iptr0 = imag + ibase ;
L47D:
		mov	edi,8[EBP]
                lea     edi,[eax*8][edi]
		mov	esi,0Ch[EBP]
                lea     esi,[eax*8][esi]
                fstp    st(0)           ; Clean stack: this is s1
                fstp    st(0)           ; And c1
		jmp	near ptr L280   ; Back to top of second loop in K2
;
;  Kernel of 4
;
;  Register usage:
;    ebx = inner_span
;    edi = rptr0
;    esi = iptr0
;
L49D:
                mov	EAX,4
		cmp	-0158h[EBP],EAX
		jne	L8B4
;     inner_span /= 4
		mov	ebx,-0170h[EBP]
		shr     ebx,2
		mov	-0170h[EBP],ebx
;     rptr0 = real ;
;     iptr0 = imag ;
                mov     edi,8[ebp]
                mov     esi,0Ch[ebp]
;     finalptr = real + jump + inner_span - inc ;
		mov	eax,-0150h[EBP]          ; jump
		add	eax,ebx                  ; inner_span
		sub	eax,-0154h[EBP]          ; inc
		lea	eax,[eax*8][edi]
		mov	-040h[EBP],eax

;  Outer loop.  FPT stack usage:
;    0 - s1
;    1 - s2
;    2 - s3
;    3 - c1
;    4 - c2
;    5 - c3

L4E5:		fld1                             ; c3
		fld1                             ; c2
		fld1                             ; c1
		fldz                             ; s3
		fldz                             ; s2
		fldz                             ; s1
;        flagptr = real + tot_pts ;
		mov	ECX,8[EBP]
		mov	EAX,-014Ch[EBP]
		lea	EAX,[EAX*8][ECX]
		mov	-04Ch[EBP],EAX
;        endptr = real + inner_span ;
		lea	eax,[ebx*8][ECX]
		mov	-044h[EBP],eax

;  Inner loop.

L52E:
;           rptr1 = rptr0 + inner_span   (eax)
;           rptr2 = rptr1 + inner_span   (ecx)
;           rptr3 = rptr2 + inner_span   (edx)
                lea     eax,[ebx*8][edi]
		mov	-068h[EBP],EAX   ; rptr1
                lea     ecx,[ebx*8][eax]
		mov	-060h[EBP],ecx   ; rptr2
                lea     edx,[ebx*8][ecx]
		mov	-058h[EBP],edx   ; rptr3

;           real_sum = *rptr0 + *rptr2
;           real_diff = *rptr0 - *rptr2
		fld	qword ptr [edi]
		fadd	qword ptr [ecx]
		fld	qword ptr [edi]
		fsub	qword ptr [ecx]
                fxch
		fstp	qword ptr -0B0h[EBP]
		fstp	qword ptr -0A8h[EBP]

;           real_sum2 = *rptr1 + *rptr3
;           real_diff2 = *rptr1 - *rptr3
		fld	qword ptr [EAX]
		fadd	qword ptr [EDX]
		fld	qword ptr [EAX]
		fsub	qword ptr [EDX]
                fxch
		fstp	qword ptr -0A0h[EBP]
		fstp	qword ptr -098h[EBP]

;           *rptr0 = real_sum + real_sum2
;           real_sum2 = real_sum - real_sum2
		fld	qword ptr -0B0h[EBP]
		fadd	qword ptr -0A0h[EBP]
		fld	qword ptr -0B0h[EBP]
		fsub	qword ptr -0A0h[EBP]
                fxch
		fstp	qword ptr [edi]
		fstp	qword ptr -0A0h[EBP]
 
;           iptr1 = iptr0 + inner_span  (eax)
;           iptr2 = iptr1 + inner_span  (ecx)
;           iptr3 = iptr2 + inner_span  (edx)
                lea     eax,[ebx*8][esi]
		mov	-064h[EBP],EAX   ; iptr1
                lea     ecx,[ebx*8][eax]
		mov	-05Ch[EBP],ecx   ; iptr2
                lea     edx,[ebx*8][ecx]
		mov	-054h[EBP],edx   ; iptr3

;           imag_sum = *iptr0 + *iptr2
;           imag_diff = *iptr0 - *iptr2
		fld	qword ptr [esi]
		fadd	qword ptr [ECX]
		fld	qword ptr [esi]
		fsub	qword ptr [ECX]
                fxch
		fstp	qword ptr -090h[EBP]
		fstp	qword ptr -088h[EBP]

;           imag_sum2 = *iptr1 + *iptr3
;           imag_diff2 = *iptr1 - *iptr3
		fld	qword ptr [EAX]
		fadd	qword ptr [EDX]
		fld	qword ptr [EAX]
		fsub	qword ptr [EDX]
                fxch
		fstp	qword ptr -080h[EBP]
		fstp	qword ptr -078h[EBP]

;           *iptr0 = imag_sum + imag_sum2
;           imag_sum2 = imag_sum - imag_sum2
		fld	qword ptr -090h[EBP]
		fadd	qword ptr -080h[EBP]
		fld	qword ptr -090h[EBP]
		fsub	qword ptr -080h[EBP]
                fxch
		fstp	qword ptr [esi]
		fstp	qword ptr -080h[EBP]
 
;          if (isign < 0)
		cmp	dword ptr 01Ch[EBP],0
		jns	L65A
		fld	qword ptr -0A8h[EBP]
		fadd	qword ptr -078h[EBP]
		fld	qword ptr -078h[EBP]
		fsubr	qword ptr -0A8h[EBP]
                fxch
		fstp	qword ptr -0B0h[EBP]
		fstp	qword ptr -0A8h[EBP]

		fld	qword ptr -088h[EBP]
		fsub	qword ptr -098h[EBP]
		fld	qword ptr -098h[EBP]
		fadd	qword ptr -088h[EBP]
                fxch
		fstp	qword ptr -090h[EBP]
		fstp	qword ptr -088h[EBP]
		jmp short	L6A0
;          else isign > 0
L65A:		fld	qword ptr -0A8h[EBP]
		fsub	qword ptr -078h[EBP]
		fld	qword ptr -078h[EBP]
		fadd	qword ptr -0A8h[EBP]
                fxch
		fstp	qword ptr -0B0h[EBP]
		fstp	qword ptr -0A8h[EBP]

		fld	qword ptr -088h[EBP]
		fadd	qword ptr -098h[EBP]
		fld	qword ptr -098h[EBP]
		fsubr	qword ptr -088h[EBP]
                fxch
		fstp	qword ptr -090h[EBP]
		fstp	qword ptr -088h[EBP]

;           *iptr1 = real_sum * s1 + imag_sum * c1
;           *iptr2 = real_sum2 * s2 + imag_sum2 * c2
;           *iptr3 = real_diff * s3 + imag_diff * c3
;           *rptr1 = real_sum * c1 - imag_sum * s1
;           *rptr2 = real_sum2 * c2 - imag_sum2 * s2
;           *rptr3 = real_diff * c3 - imag_diff * s3
;   Note order reversed relative to C so we can use eax, ecx, edx for iptr's

L6A0:
		fld	qword ptr -0B0h[EBP]     ; real_sum
		fmul	st,st(1)                 ; s1
		fld	qword ptr -090h[EBP]     ; imag_sum
		fmul	st,st(5)                 ; c1
		faddp	ST(1),ST
		fstp	qword ptr [eax]          ; *iptr1

		fld	qword ptr -0A0h[EBP]     ; real_sum2
		fmul	st,st(2)                 ; s2
		fld	qword ptr -080h[EBP]     ; imag_sum2
		fmul	st,st(6)                 ; c2
		faddp	ST(1),ST
		fstp	qword ptr [ecx]          ; *iptr2

		fld	qword ptr -0A8h[EBP]     ; real_diff
		fmul	st,st(3)                 ; s3
		fld	qword ptr -088h[EBP]     ; imag_diff
		fmul	st,st(7)                 ; c3
		faddp	ST(1),ST
		fstp	qword ptr [edx]          ; *iptr3

                fld	qword ptr -0B0h[EBP]     ; real_sum
		fmul	st,st(4)                 ; c1
		fld	qword ptr -090h[EBP]
		fmul	st,st(2)                 ; s1
		fsubp	ST(1),ST
		mov	EAX,-068h[EBP]
		fstp	qword ptr [EAX]          ; *rptr1

		fld	qword ptr -0A0h[EBP]     ; real_sum2
		fmul	st,st(5)                 ; c2
		fld	qword ptr -080h[EBP]     ; imag_sum2
		fmul	st,st(3)                 ; s2
		fsubp	ST(1),ST
		mov	EAX,-060h[EBP]
		fstp	qword ptr [EAX]          ; *rptr2

		fld	qword ptr -0A8h[EBP]     ; real_diff
		fmul	st,st(6)                 ; c3
		fld	qword ptr -088h[EBP]     ; imag_diff
		fmul	st,st(4)                 ; s3
		fsubp	ST(1),ST
		mov	EAX,-058h[EBP]
		fstp	qword ptr [EAX]          ; *rptr3

;           rptr0 += 4 * inner_span
;           iptr0 += 4 * inner_span
		mov	eax,ebx
		shl	eax,5
                add     edi,eax
                add     esi,eax

;           if (rptr0 < flagptr)
		cmp	edi,-04Ch[EBP]
		jb	near ptr L52E    ; Continue

;           rptr0 += jump - tot_pts
;           iptr0 += jump - tot_pts
		mov	EAX,-0150h[EBP]
		sub	EAX,-014Ch[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax

;           if (rptr0 >= endptr)
		cmp	edi,-044h[EBP]
		jae	near ptr L879     ; Break out of inner loop

;  Trig update

;           temp = c1 - (c0 * c1 + s0 * s1) ;
		fld	qword ptr -0148h[EBP]    ; c0
		fmul	st,st(4)                 ; c1
		fld	qword ptr -0140h[EBP]    ; s0
		fmul	st,st(2)                 ; s1
		faddp	ST(1),ST                 ; Sum now in sp(0)
                fstp    st(6)                    ; Save sum in c3 spot

;           s1 = (s0 * c1 - c0 * s1) + s1 ;
		fld	qword ptr -0140h[EBP]    ; s0
		fmul	st,st(4)                 ; c1
		fld	qword ptr -0148h[EBP]    ; c0
		fmul	st,st(2)                 ; s1
		fsubp	ST(1),ST
		fadd	st,st(1)                 ; s1
                fstp    st(1)                    ; s1

;           c1 = temp
                fld	st(5)                    ; Sum was saved in c3 spot
		fsubp	ST(4),ST                 ; Subtract it from c1

;           c2 = c1 * c1 - s1 * s1
		fld	st(3)                    ; c1
		fmul	st,st(4)                 ; c1
		fld	st(1)                    ; s1
		fmul	st,st(2)                 ; s1
		fsubp	ST(1),ST
                fstp    st(5)                    ; c2

;           s2 = 2.0 * c1 * s1
		fld	st(3)                    ; c1
		fmul	st,st(1)                 ; s1
                fadd    st,st                    ; Double it
                fstp    st(2)                    ; s2

;           c3 = c2 * c1 - s2 * s1 ;
		fld	st(4)                    ; c2
		fmul	st,st(4)                 ; c1
		fld	st(2)                    ; s2
		fmul	st,st(2)                 ; s1
		fsubp	ST(1),ST
                fstp    st(6)                    ; c3

;           s3 = c2 * s1 + s2 * c1 ;
		fld	st(4)                    ; c2
		fmul	st,st(1)                 ; s1
		fld	st(2)                    ; s2
		fmul	st,st(5)                 ; c1
		faddp	ST(1),ST
                fstp    st(3)                    ; s3
		jmp	near ptr L52E   ; Back to top of inner loop

L879:
;   Done with 6 trig functions for now.  Clean up FPT stack.                
                fstp    st(0)
                fstp    st(0)
                fstp    st(0)
                fstp    st(0)
                fstp    st(0)
                fstp    st(0)

;        if (rptr0 >= finalptr)
		cmp	edi,-040h[EBP]
		jae     short	L89D

;        rptr0 += inc - inner_span
;        iptr0 += inc - inner_span
		mov	EAX,-0154h[EBP]
		sub	eax,ebx
		shl	EAX,3
                add     edi,eax
                add     esi,eax
		jmp	near ptr L4E5   ; Back to top of outer loop

;     if (inner_span == jump)
;        return ;
L89D:		cmp	ebx,-0150h[EBP]
		jne	LD4
		pop	EDI
		pop	ESI
		pop	EBX
		leave
		ret
;
;  All odd kernels
;
L8B4:
;     full_span = inner_span
;     inner_span /= kernel
                mov	EAX,-0170h[EBP]
		mov	-016Ch[EBP],EAX
		mov	ECX,-0158h[EBP]
		mov	EAX,-0170h[EBP]
		cdq
		idiv	ECX
		mov	-0170h[EBP],EAX
;     if (kernel == 3)
		mov	EDX,3
		cmp	ECX,EDX
		jne	LA52
;
;  Kernel of 3
;
;  Register usage:
;    edi - rptr0
;    esi - iptr0
               
;        rptr0 = real
;        iptr0 = imag
		mov	edi,8[EBP]
		mov	esi,0Ch[EBP]

;        flagptr = real + last_point - 1
;        endptr = flagptr + inner_span + 1
		mov	eax,-0160h[EBP]          ; last_point
                dec     eax
		lea	eax,[eax*8][edi]
		mov	-04Ch[EBP],eax

                mov	ebx,-0170h[EBP]          ; inner_span
                inc     ebx
		lea	eax,[ebx*8][eax]
		mov	-044h[EBP],EAX

;        for (;;) {
L906:
;           rptr1 = rptr0 + inner_span
;           rptr2 = rptr1 + inner_span
                mov	ebx,-0170h[EBP]          ; inner_span
                lea     eax,[ebx*8][edi]
                lea     ecx,[ebx*8][eax]

;           rtemp = *rptr0
		fld	qword ptr [edi]
		fstp	qword ptr -0D8h[EBP]

;           real_sum = *rptr1 + *rptr2
		fld	qword ptr [EAX]
		fadd	qword ptr [ECX]
		fst	qword ptr -0B0h[EBP]

;           *rptr0 = rtemp + real_sum
		fadd	qword ptr -0D8h[EBP]
		fstp	qword ptr [edi]

;           rtemp -= 0.5 * real_sum
;           rtemp2 = (*rptr1 - *rptr2) * sin_third
		fld1
                fadd    st,st
		fld	qword ptr -0B0h[EBP]     ; real_sum
                fdivrp  st(1),st
		fsubr	qword ptr -0D8h[EBP]     ; rtemp
		fld	qword ptr [EAX]          ; *rptr1
		fsub	qword ptr [ECX]          ; *rptr2
		fmul	qword ptr -0F8h[EBP]     ; sin_third
                fxch
		fstp	qword ptr -0D8h[EBP]     ; rtemp
		fstp	qword ptr -0C8h[EBP]     ; rtemp2
 
;           iptr1 = iptr0 + inner_span
;           iptr2 = iptr1 + inner_span
                mov	ebx,-0170h[EBP]          ; inner_span
                lea     edx,[ebx*8][esi]
                lea     ebx,[ebx*8][edx]

;           itemp = *iptr0
		fld	qword ptr [esi]
		fstp	qword ptr -0D0h[EBP]

;           imag_sum = *iptr1 + *iptr2
;           *iptr0 = itemp + imag_sum
		fld	qword ptr [EDX]
		fadd	qword ptr [EBX]
		fst	qword ptr -090h[EBP]
		fadd	qword ptr -0D0h[EBP]
		fstp	qword ptr [esi]

;           itemp -= 0.5 * imag_sum
;           itemp2 = (*iptr1 - *iptr2) * sin_third

		fld1
                fadd    st,st
		fld	qword ptr -090h[EBP]     ; imag_sum
                fdivrp  st(1),st
		fsubr	qword ptr -0D0h[EBP]     ; itemp
		fld	qword ptr [EDX]          ; *iptr1
		fsub	qword ptr [EBX]          ; *iptr2
		fmul	qword ptr -0F8h[EBP]     ; sin_third
                fxch
		fstp	qword ptr -0D0h[EBP]     ; itemp
		fstp	qword ptr -0C0h[EBP]     ; itemp2

;           *rptr1 = rtemp - itemp2
;           *rptr2 = rtemp + itemp2
		fld	qword ptr -0D8h[EBP]
		fsub	qword ptr -0C0h[EBP]
		fld	qword ptr -0D8h[EBP]
		fadd	qword ptr -0C0h[EBP]
                fxch
		fstp	qword ptr [EAX]
		fstp	qword ptr [ECX]

;           *iptr1 = itemp + rtemp2
;           *iptr2 = itemp - rtemp2
		fld	qword ptr -0D0h[EBP]
		fadd	qword ptr -0C8h[EBP]
		fld	qword ptr -0D0h[EBP]
		fsub	qword ptr -0C8h[EBP]
                fxch
		fstp	qword ptr [EDX]
		fstp	qword ptr [EBX]
 
;           rptr0 += 3 * inner_span
;           iptr0 += 3 * inner_span
		mov	EAX,-0170h[EBP]
		imul	EAX,EAX,018h
                add     edi,eax
                add     esi,eax

;           if (rptr0 < flagptr)
;              continue
		cmp	edi,-04Ch[EBP]
		jb	near ptr L906   ; Continue
;           if (rptr0 >= endptr)
;              break
		cmp	edi,-044h[EBP]
		jae	near ptr L11CA  ; Break

;           rptr0 -= last_point
;           iptr0 -= last_point
                mov	EAX,-0160h[EBP]
		shl	EAX,3
		sub	edi,EAX
		sub	esi,EAX
		jmp	near ptr L906   ; Back to top of loop
;
;  Kernel of 5
;
LA52:		cmp	dword ptr -0158h[EBP],5
		jne	LD82
;        c2 = cos_fifth * cos_fifth - sin_fifth * sin_fifth
;        s2 = 2.0 * cos_fifth * sin_fifth
		fld	qword ptr -0E8h[EBP]
		fmul	qword ptr -0E8h[EBP]
		fld	qword ptr -0F0h[EBP]
		fmul	qword ptr -0F0h[EBP]
		fsubp	ST(1),ST
		fstp	qword ptr -0128h[EBP]
		fld	qword ptr -0E8h[EBP]
		fmul	qword ptr -0F0h[EBP]
                fadd    st,st
		fstp	qword ptr -0120h[EBP]

;        rptr0 = real
;        iptr0 = imag
		mov	EAX,8[EBP]
		mov	-070h[EBP],EAX
		mov	ECX,0Ch[EBP]
		mov	-06Ch[EBP],ECX

;        flagptr = real + last_point - 1
;        endptr = flagptr + inner_span + 1
		mov	ECX,-0160h[EBP]
		lea	EAX,[ECX*8][EAX]
		add	EAX,0FFFFFFF8h
		mov	-04Ch[EBP],EAX
		mov	ECX,-0170h[EBP]
		lea	EAX,[ECX*8][EAX]
		add	EAX,8
		mov	-044h[EBP],EAX

;        for (;;) {
LAC3:
;           rptr1 = rptr0 + inner_span
;           rptr2 = rptr1 + inner_span
;           rptr3 = rptr2 + inner_span
;           rptr4 = rptr3 + inner_span
                mov	EAX,-0170h[EBP]
		shl	EAX,3
		mov	-028h[EBP],EAX
		add	EAX,-070h[EBP]
		mov	-068h[EBP],EAX  ; rptr1 in EAX
		mov	ECX,-028h[EBP]
		add	ECX,EAX
		mov	-060h[EBP],ECX  ; rptr2 in ECX
		mov	EDX,-028h[EBP]
		add	EDX,ECX
		mov	-058h[EBP],EDX  ; rptr3 in EDX
		mov	EBX,-028h[EBP]
		add	EBX,EDX         ; rptr4 in EBX

;           real_sum = *rptr1 + *rptr4
;           real_diff = *rptr1 - *rptr4
		fld	qword ptr [EAX]
		fadd	qword ptr [EBX]
		fld	qword ptr [EAX]
		fsub	qword ptr [EBX]
                fxch
		fstp	qword ptr -0B0h[EBP]
		fstp	qword ptr -0A8h[EBP]

;           real_sum2 = *rptr2 + *rptr3
;           real_diff2 = *rptr2 - *rptr3
		fld	qword ptr [ECX]
		fadd	qword ptr [EDX]
		fld	qword ptr [ECX]
		fsub	qword ptr [EDX]
                fxch
		fstp	qword ptr -0A0h[EBP]
		fstp	qword ptr -098h[EBP]

;           r0 = *rptr0
		mov	edi,-070h[EBP]
		fld	qword ptr [edi]
		fst	qword ptr -0108h[EBP]

;           *rptr0 = r0 + real_sum + real_sum2
		fadd	qword ptr -0B0h[EBP]
		fadd	qword ptr -0A0h[EBP]
		fstp	qword ptr [edi]

;           rtemp = real_sum * cos_fifth + real_sum2 * c2 + r0
		fld	qword ptr -0B0h[EBP]
		fmul	qword ptr -0E8h[EBP]
		fld	qword ptr -0A0h[EBP]
		fmul	qword ptr -0128h[EBP]
                fxch
		fadd	qword ptr -0108h[EBP]
		faddp	ST(1),ST
		fstp	qword ptr -0D8h[EBP]
 
;           iptr1 = iptr0 + inner_span
;           iptr2 = iptr1 + inner_span
;           iptr3 = iptr2 + inner_span
;           iptr4 = iptr3 + inner_span
		mov	ESI,-028h[EBP]
		add	ESI,-06Ch[EBP]
		mov	-064h[EBP],ESI           ; iptr1 in ESI
		mov	EDI,-028h[EBP]
		add	EDI,ESI
		mov	-05Ch[EBP],EDI           ; iptr2 in EDI
		mov	EAX,-028h[EBP]
		add	EAX,EDI
		mov	-054h[EBP],EAX           ; iptr3
		mov	EAX,-028h[EBP]
		add	EAX,-054h[EBP]
		mov	-050h[EBP],EAX           ; iptr4 in EAX

;           imag_sum = *iptr1 + *iptr4
;           imag_diff = *iptr1 - *iptr4
		fld	qword ptr [ESI]
		fadd	qword ptr [EAX]
		fld	qword ptr [ESI]
		fsub	qword ptr [EAX]
                fxch
		fstp	qword ptr -090h[EBP]
		fstp	qword ptr -088h[EBP]

;           imag_sum2 = *iptr2 + *iptr3
;           imag_diff2 = *iptr2 - *iptr3
		mov	EAX,-054h[EBP]
		fld	qword ptr [EDI]
		fadd	qword ptr [EAX]
		fld	qword ptr [EDI]
		fsub	qword ptr [EAX]
                fxch
		fstp	qword ptr -080h[EBP]
		fstp	qword ptr -078h[EBP]
 
;           i0 = *iptr0
;           *iptr0 = i0 + imag_sum + imag_sum2
		mov	EAX,-06Ch[EBP]
		fld	qword ptr [EAX]
		fst	qword ptr -0100h[EBP]
		fadd	qword ptr -090h[EBP]
		fadd	qword ptr -080h[EBP]
		fstp	qword ptr [EAX]

;           itemp = imag_sum * cos_fifth + imag_sum2 * c2 + i0
		fld	qword ptr -090h[EBP]
		fmul	qword ptr -0E8h[EBP]
		fld	qword ptr -080h[EBP]
		fmul	qword ptr -0128h[EBP]
                fxch
		fadd	qword ptr -0100h[EBP]
		faddp	ST(1),ST
		fstp	qword ptr -0D0h[EBP]

;           rtemp2 = real_diff * sin_fifth + real_diff2 * s2
;           itemp2 = imag_diff * sin_fifth + imag_diff2 * s2
		fld	qword ptr -0A8h[EBP]
		fmul	qword ptr -0F0h[EBP]
		fld	qword ptr -098h[EBP]
		fmul	qword ptr -0120h[EBP]
		faddp	ST(1),ST
		fld	qword ptr -088h[EBP]
		fmul	qword ptr -0F0h[EBP]
		fld	qword ptr -078h[EBP]
		fmul	qword ptr -0120h[EBP]
		faddp	ST(1),ST
                fxch
		fstp	qword ptr -0C8h[EBP]
		fstp	qword ptr -0C0h[EBP]

;           *rptr1 = rtemp - itemp2
;           *rptr4 = rtemp + itemp2
		fld	qword ptr -0D8h[EBP]
		fsub	qword ptr -0C0h[EBP]
		fld	qword ptr -0D8h[EBP]
		fadd	qword ptr -0C0h[EBP]
                fxch
		mov	EAX,-068h[EBP]
		fstp	qword ptr [EAX]
		fstp	qword ptr [EBX]

;           *iptr1 = itemp + rtemp2
;           *iptr4 = itemp - rtemp2
		fld	qword ptr -0D0h[EBP]
		fadd	qword ptr -0C8h[EBP]
		fld	qword ptr -0D0h[EBP]
		fsub	qword ptr -0C8h[EBP]
                fxch
		fstp	qword ptr [ESI]
		mov	EAX,-050h[EBP]
		fstp	qword ptr [EAX]

;           rtemp = real_sum * c2 + real_sum2 * cos_fifth + r0
		fld	qword ptr -0B0h[EBP]
		fmul	qword ptr -0128h[EBP]
		fld	qword ptr -0A0h[EBP]
		fmul	qword ptr -0E8h[EBP]
                fxch
		fadd	qword ptr -0108h[EBP]
		faddp	ST(1),ST
		fstp	qword ptr -0D8h[EBP]

;           itemp = imag_sum * c2 + imag_sum2 * cos_fifth + i0
		fld	qword ptr -090h[EBP]
		fmul	qword ptr -0128h[EBP]
		fld	qword ptr -080h[EBP]
		fmul	qword ptr -0E8h[EBP]
                fxch
		fadd	qword ptr -0100h[EBP]
		faddp	ST(1),ST
		fstp	qword ptr -0D0h[EBP]

;           rtemp2 = real_diff * s2 - real_diff2 * sin_fifth
;           itemp2 = imag_diff * s2 - imag_diff2 * sin_fifth
		fld	qword ptr -0A8h[EBP]
		fmul	qword ptr -0120h[EBP]
		fld	qword ptr -098h[EBP]
		fmul	qword ptr -0F0h[EBP]
		fsubp	ST(1),ST

		fld	qword ptr -088h[EBP]
		fmul	qword ptr -0120h[EBP]
		fld	qword ptr -078h[EBP]
		fmul	qword ptr -0F0h[EBP]
		fsubp	ST(1),ST

                fxch
		fstp	qword ptr -0C8h[EBP]
		fstp	qword ptr -0C0h[EBP]

;           *rptr2 = rtemp - itemp2
;           *rptr3 = rtemp + itemp2
		fld	qword ptr -0D8h[EBP]
		fsub	qword ptr -0C0h[EBP]
		fld	qword ptr -0D8h[EBP]
		fadd	qword ptr -0C0h[EBP]
                fxch
		fstp	qword ptr [ECX]
		fstp	qword ptr [EDX]

;           *iptr2 = itemp + rtemp2
;           *iptr3 = itemp - rtemp2
		fld	qword ptr -0D0h[EBP]
		fadd	qword ptr -0C8h[EBP]
		fld	qword ptr -0D0h[EBP]
		mov	EAX,-054h[EBP]
		fsub	qword ptr -0C8h[EBP]
                fxch
		fstp	qword ptr [EDI]
		fstp	qword ptr [EAX]

;           rptr0 += 5 * inner_span
;           iptr0 += 5 * inner_span
		mov	EAX,-0170h[EBP]
		imul	EAX,EAX,028h
		add	-070h[EBP],EAX
		add	-06Ch[EBP],EAX

;           if (rptr0 < flagptr)
;              continue
		mov	EAX,-070h[EBP]
		cmp	EAX,-04Ch[EBP]
		jb	near ptr LAC3

;           if (rptr0 >= endptr)
;              break
		mov	EAX,-070h[EBP]
		cmp	EAX,-044h[EBP]
		jae	near ptr L11CA

;           rptr0 -= last_point
;           iptr0 -= last_point
		mov	EAX,-0160h[EBP]
		shl	EAX,3
		sub	-070h[EBP],EAX
		sub	-06Ch[EBP],EAX
		jmp	near ptr LAC3
;
;  Arbitrary odd kernel
;
LD82:		mov	EAX,-0158h[EBP]
		cmp	EAX,-015Ch[EBP]
		je	LEA8
; Compute new trig.  Register usage:
;  EDI - rptr0
;  ESI - iptr0
;  ECX - rptr1
;  EDX - iptr1
;
;           angle = 2.0 * PI / kernel
		mov	-015Ch[EBP],EAX
		fldpi
                fadd    st,st
		fidiv	dword ptr -0158h[EBP]
		fstp	qword ptr -0E0h[EBP]

;           if (isign < 0)
;              angle = -angle
		cmp	dword ptr 01Ch[EBP],0
		jns	LDC2
		fld	qword ptr -0E0h[EBP]
		fchs
		fstp	qword ptr -0E0h[EBP]

;           c1 = cos ( angle )
;           s1 = sin ( angle )
LDC2:
		fld	qword ptr -0E0h[EBP]   ; angle
                fsincos
		fst	qword ptr -0138h[EBP]  ; c1 (Keep in st(1))
                fxch
		fst	qword ptr -0130h[EBP]  ; s1 (Keep in st(0))

;           rptr0 = cosines
;           rptr1 = cosines + kernel - 1
;           iptr0 = sines
;           iptr1 = sines + kernel - 1

		mov	eax,-0158h[EBP]          ; kernel
		dec	eax
		mov	edi,02Ch[EBP]            ; rptr0
		mov	esi,030h[EBP]            ; iptr0
                lea     ecx,[eax*8][edi]         ; rptr1
                lea     edx,[eax*8][esi]         ; iptr1

;           cosines[kernel-1] = 1.0 ;
;           sines[kernel-1] = 0.0
		fld1
		fst	qword ptr [ecx]
		fldz
		fst	qword ptr [edx]

;           for (;;) {
LE43:
;              *rptr0 = *rptr1 * c1 + *iptr1 * s1
                fld     st(1)                    ; Previous rptr
                fmul    st,st(4)                 ; c1
                fld     st(1)                    ; Previous iptr
                fmul    st,st(4)                 ; s1
		faddp	ST(1),ST
		fst	qword ptr [edi]

;              *iptr0 = *rptr1 * s1 - *iptr1 * c1
                fld     st(2)                    ; Previous rptr
                fmul    st,st(4)                 ; s1
                fld     st(2)                    ; Previous iptr
                fmul    st,st(6)                 ; c1
		fsubp	ST(1),ST
		fst	qword ptr [esi]
                fxch
 
;              *(--rptr1) = *(rptr0++)
                add     edi,8
                fst     st(3)                    ; save *rptr
                sub     ecx,8
		fstp	qword ptr [ecx]
 
;              *(--iptr1) = -*(iptr0++)
                add     esi,8
		fchs
                sub     edx,8
                fst     st(1)                    ; save *iptr
		fstp	qword ptr [edx]
 
;              if (rptr0 >= rptr1)
;                 break
		cmp	edi,ecx
		jb	LE43

                fstp    st(0)                    ; iptr
                fstp    st(0)                    ; rptr
                fstp    st(0)                    ; s1
                fstp    st(0)                    ; c1

;
;  Trig is ready.  Do the odd kernel.
;
;  Register usage:
;    ebx = inner_span
;    edi = rptr0
;    esi = iptr0

LEA8:
;        rptr0 = real
;        iptr0 = imag
		mov	ebx,-0170h[EBP]
                mov	edi,8[EBP]
		mov	esi,0Ch[EBP]

;        flagptr = real + last_point
;        endptr = flagptr + inner_span
		mov	ECX,-0160h[EBP]          ; last_point
		lea	EAX,[ECX*8][edi]
		mov	-04Ch[EBP],EAX
		lea	EAX,[ebx*8][EAX]
		mov	-044h[EBP],EAX

;        for (;;) {
LECC:
;           rptr1 = rptr0 + inner_span
                lea     eax,[ebx*8][edi]
		mov	-068h[EBP],EAX
;           rptr2 = rptr0 + full_span
		mov	EAX,-016Ch[EBP]
		shl	EAX,3
		mov	-01Ch[EBP],EAX
		add	EAX,edi
		mov	-060h[EBP],EAX

;           rtemp = r0 = *rptr0
		fld	qword ptr [edi]
		fst	qword ptr -0108h[EBP]
		fstp	qword ptr -0D8h[EBP]
 
;           iptr1 = iptr0 + inner_span
;           iptr2 = iptr0 + full_span
		lea	eax,[ebx*8][esi]
		mov	-064h[EBP],EAX
		mov	EAX,-01Ch[EBP]
		add	EAX,esi
		mov	-05Ch[EBP],EAX

;           itemp = i0 = *iptr0
		fld	qword ptr [esi]
		fst	qword ptr -0100h[EBP]
		fstp	qword ptr -0D0h[EBP]
 
;  First middle loop.  Register usage:
;    edi - rptr3
;    esi - iptr3   (Pushed at start, popped at end to preserve)

                push   edi
                push   esi
;           rptr3 = rwork
;           iptr3 = iwork
		mov	edi,024h[EBP]   ; rwork
		mov	esi,028h[EBP]   ; iwork

;           for (;;) {
LF34:
;              rptr2 -= inner_span
                mov	EAX,ebx
		shl	EAX,3
		sub	-060h[EBP],EAX

;              *rptr3 = *rptr1 + *rptr2
;              rtemp += *(rptr3++)
;              *(rptr3++) = *rptr1 - *rptr2
		mov	ECX,-068h[EBP]           ; rptr1
		fld	qword ptr [ECX]
		mov	EDX,-060h[EBP]           ; rptr2
		fadd	qword ptr [EDX]
		fst	qword ptr [edi]
                add     edi,8

		fadd	qword ptr -0D8h[EBP]     ; rtemp
		fld	qword ptr [ECX]          ; rptr1
		fsub	qword ptr [EDX]          ; rptr2
                fxch
		fstp	qword ptr -0D8h[EBP]     ; rtemp
		fstp	qword ptr [edi]          ; rptr3
                add     edi,8

;              rptr1 += inner_span
;              iptr2 -= inner_span
                mov	EAX,ebx
		shl	EAX,3
		add	-068h[EBP],EAX
		sub	-05Ch[EBP],EAX

;              *iptr3 = *iptr1 + *iptr2
;              itemp += *(iptr3++)
;              *(iptr3++) = *iptr1 - *iptr2
		mov	ECX,-064h[EBP]
		fld	qword ptr [ECX]
		mov	edx,-05Ch[EBP]
		fadd	qword ptr [edx]
		fst	qword ptr [esi]
                add     esi,8

		fadd	qword ptr -0D0h[EBP]
		fld	qword ptr [ECX]
		fsub	qword ptr [edx]
                fxch
		fstp	qword ptr -0D0h[EBP]
		fstp	qword ptr [esi]
                add     esi,8
 
;              iptr1 += inner_span
                mov     eax,ebx
                shl     eax,3
		add	-064h[EBP],EAX

;              if (rptr1 >= rptr2)
;                 break
                mov     edx,-060h[ebp]
		cmp	-068h[EBP],EDX
		jb	LF34  ; Back to top of first middle loop

                pop     esi   ; Restore (saved at start of loop)
                pop     edi

; End of first middle loop.  Continue outer loop.

;           *rptr0 = rtemp
;           *iptr0 = itemp
		fld	qword ptr -0D8h[EBP]
		fstp	qword ptr [edi]
		fld	qword ptr -0D0h[EBP]
		fstp	qword ptr [esi]
 
;           rptr1 = rptr0 + inner_span
;           rptr2 = rptr0 + full_span
                lea     eax,[ebx*8][edi]
		mov	-068h[EBP],eax
		mov	edx,-016Ch[EBP]  ; full_span
                lea     eax,[edx*8][edi]
		mov	-060h[EBP],eax

;           iptr1 = iptr0 + inner_span
;           iptr2 = iptr0 + full_span
                lea     eax,[ebx*8][esi]
		mov	-064h[EBP],EAX
                lea     eax,[edx*8][esi]
		mov	-05Ch[EBP],EAX

;           j = 1
		mov	dword ptr -0184h[EBP],1

;  Second middle loop

;           for (;;) {
L1016:
;              rtemp = r0
;              itemp = i0
                fld	qword ptr -0108h[EBP]
		fstp	qword ptr -0D8h[EBP]
		fld	qword ptr -0100h[EBP]
		fstp	qword ptr -0D0h[EBP]

;              rtemp2 = itemp2 = 0.0
		fldz
		fst	qword ptr -0C0h[EBP]
		fstp	qword ptr -0C8h[EBP]
 
;              finalptr = rwork + kernel - 1
		mov	EAX,024h[EBP]            ; rwork
		mov	ECX,-0158h[EBP]          ; kernel
                dec     ecx
		lea	EAX,[ECX*8][EAX]
		mov	-040h[EBP],EAX

;              itrig = j - 1
		mov	EAX,-0184h[EBP]
		dec	EAX
		mov	-0164h[EBP],EAX

;   Innermost loop... Register usage:
;     edi - rptr3
;     esi - iptr3  (Push at start, pop at end to preserve them)
;     ecx - cosines[itrig]
;     edx - sines[itrig]

                push    edi
                push    esi
;              rptr3 = rwork
;              iptr3 = iwork
		mov	edi,024h[EBP]            ; rwork
		mov	esi,028h[EBP]            ; iwork
		mov	edx,-0164h[EBP]          ; itrig
;              for (;;) {
L1068:          ; Note that itrig is still in edx from bottom of loop
		mov	eax,edx                  ; itrig (see loop bottom)
		mov	ecx,02Ch[EBP]            ; cosines
		mov	edx,030h[EBP]            ; sines
		lea     ecx,[eax*8][ecx]         ; cosines[itrig]
		lea     edx,[eax*8][edx]         ; sines[itrig]

;                 rtemp += *(rptr3++) * cosines[itrig]
;                 rtemp2 += *(rptr3++) * sines[itrig]
		fld	qword ptr [edi]
		fmul	qword ptr [ecx]
		add	edi,8
		fadd	qword ptr -0D8h[EBP]
 
		fld	qword ptr [edi]
		fmul	qword ptr [edx]
		add	edi,8
		fadd	qword ptr -0C8h[EBP]

                fxch
		fstp	qword ptr -0D8h[EBP]
		fstp	qword ptr -0C8h[EBP]
 
;                 itemp += *(iptr3++) * cosines[itrig]
;                 itemp2 += *(iptr3++) * sines[itrig]
		fld	qword ptr [esi]
		fmul	qword ptr [ecx]
		add	esi,8
		fadd	qword ptr -0D0h[EBP]
 
		fld	qword ptr [esi]
		fmul	qword ptr [edx]
		add	esi,8
		fadd	qword ptr -0C0h[EBP]

                fxch
		fstp	qword ptr -0D0h[EBP]
		fstp	qword ptr -0C0h[EBP]
 
;                 if (rptr3 >= finalpt
;                    break
		cmp	edi,-040h[EBP]
		jae     short	L110D  ; Break

;                 itrig = (itrig + j) % kernel
L10EF:
                mov	EAX,-0164h[EBP]
		add	EAX,-0184h[EBP]
		cdq
		idiv	dword ptr -0158h[EBP]
		mov	-0164h[EBP],EDX
		jmp	near ptr L1068   ; Back to top of inner loop

L110D:
            
                pop     esi    ; Restore from start of innnermost loop
                pop     edi

;  Continue second middle loop

;              rptr2 -= inner_span
;              iptr2 -= inner_span
                mov     eax,ebx
		shl	EAX,3
		sub	-060h[EBP],EAX
		sub	-05Ch[EBP],EAX

;              *rptr1 = rtemp - itemp2
;              *rptr2 = rtemp + itemp2
		fld	qword ptr -0D8h[EBP]
		fsub	qword ptr -0C0h[EBP]
		fld	qword ptr -0D8h[EBP]
		fadd	qword ptr -0C0h[EBP]
                fxch
		mov	EAX,-068h[EBP]
		fstp	qword ptr [EAX]
		mov	EAX,-060h[EBP]
		fstp	qword ptr [EAX]

;              *iptr1 = itemp + rtemp2
;              *iptr2 = itemp - rtemp2
		fld	qword ptr -0D0h[EBP]
		fadd	qword ptr -0C8h[EBP]
		fld	qword ptr -0D0h[EBP]
		fsub	qword ptr -0C8h[EBP]
                fxch
		mov	EAX,-064h[EBP]
		fstp	qword ptr [EAX]
		mov	EAX,-05Ch[EBP]
		fstp	qword ptr [EAX]
 
;              if (2 * (++j) > kernel)
;                 break
		inc	dword ptr -0184h[EBP]
		mov	EAX,-0184h[EBP]
		shl	EAX,1
		cmp	EAX,-0158h[EBP]
		jg      short	L1190     ; Break out

;              rptr1 += inner_span
;              iptr1 += inner_span
		mov	EAX,ebx
		shl	EAX,3
		add	-068h[EBP],EAX
		add	-064h[EBP],EAX
		jmp	near ptr L1016   ; Back to top of middle loop

;           rptr0 += full_span
;           iptr0 += full_span
L1190:		mov	EAX,-016Ch[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax

;           if (rptr0 < flagptr)
;              continue
		cmp	edi,-04Ch[EBP]
		jb	near ptr LECC

;           if (rptr0 >= endptr)
;              break
		cmp	edi,-044h[EBP]
		jae     short	L11CA

;           rptr0 -= last_point
;           iptr0 -= last_point
L11B6:		mov	EAX,-0160h[EBP]
		shl	EAX,3
                sub     edi,eax
                sub     esi,eax
		jmp	near ptr LECC   ; Back to top of outer loop

;     if (facnum == n_facs)
;        return
L11CA:		mov	EAX,-0168h[EBP]
		cmp	EAX,020h[EBP]
		jne	L11DA
		pop	EDI
		pop	ESI
		pop	EBX
		leave
		ret
;
;  Multiply by rotation factor
;
L11DA:
;     rptr0 = real + jump
;     iptr0 = imag + jump
                mov	EAX,-0150h[EBP]
		mov	edi,8[EBP]
		mov	esi,0Ch[EBP]
                lea     edi,[eax*8][edi]
                lea     esi,[eax*8][esi]

;     j = inner_span - tot_pts
		mov	EAX,-0170h[EBP]
		sub	EAX,-014Ch[EBP]
		mov	-0184h[EBP],EAX

;     k = full_span + tot_pts - jump
		mov	ECX,-016Ch[EBP]
		add	ECX,-014Ch[EBP]
		sub	ECX,-0150h[EBP]
		mov	-0180h[EBP],ECX

;     l = inner_span - k
		mov	EDX,-0170h[EBP]
		sub	EDX,ECX
		mov	-017Ch[EBP],EDX

;     m = jump + inc - k
		mov	EDX,-0150h[EBP]
		add	EDX,-0154h[EBP]
		sub	EDX,ECX
		mov	-0178h[EBP],EDX

;     flagptr = real + tot_pts
;     flagptr2 = real + k
		mov	ebx,8[EBP]        ; real
		mov	edx,-014Ch[EBP]   ; tot_pts
                lea     edx,[edx*8][ebx]
		mov	-04Ch[EBP],EDX    ; flagptr

		lea	edx,[ecx*8][ebx]  ; ecx holds k
		mov	-048h[EBP],edx    ; flagptr2

;     endptr = real + full_span - j
		mov	ECX,-016Ch[EBP]   ; full_span
                sub     ecx,eax           ; j in eax
                lea     edx,[ecx*8][ebx]
		mov	-044h[EBP],edx    ; endptr

;     finalptr = real + full_span + tot_pts - inc
		mov	ECX,-016Ch[EBP]   ; full_span
                add     ecx,-014Ch[ebp]   ; tot_pts
                sub     ecx,-0154h[ebp]   ; inc
                lea     edx,[ecx*8][ebx]
		mov	-040h[EBP],edx  ; finalptr

;  Outer loop

;  We push c0 and s0 to keep them available in the stack
;  At the start of each pass through this loop, also push c2 and s1.
;  Pop them at the bottom of the loop to keep it clean.
;  Overall stack usage:
;    0 - s2
;    1 - c1
;    2 - s1
;    3 - c2
;    4 - s0
;    5 - c0


 		fld	qword ptr -0148h[EBP]  ; c0
		fld	qword ptr -0140h[EBP]  ; s0

L1288:
;         c2 = 1.0 - c0
;         s1 = s0
                fld1
		fsub	qword ptr -0148h[EBP]   ; c2 = 1 - c0
		fld	qword ptr -0140h[EBP]   ; s1 = s0
                fldz                            ; c1 placeholder
                fldz                            ; s2 placeholder

;  Middle loop

L12A4:
;           c1 = c2
;           s2 = s1
                fld	st(3)                   ; c2
                fstp    st(2)                   ; c1
		fld	st(2)                   ; s1
                fstp    st(1)                   ; s2

;           rptr0 += inner_span ;
;           iptr0 += inner_span ;
		mov	EAX,-0170h[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax

;  Inner loop

L12CD:
;              rtemp = *rptr0
		fld	qword ptr [edi]
		fst	qword ptr -0D8h[EBP]  ; rtemp

;              *rptr0 = c2 * rtemp - s2 * *iptr0
		fmul	st,st(4)              ; c2
		fld	qword ptr [esi]
		fmul	st,st(2)              ; s2
		fsubp	ST(1),ST
		fstp	qword ptr [edi]       ; *rptr0

;              *iptr0 = s2 * rtemp + c2 * *iptr0
		fld	qword ptr -0D8h[EBP]  ; rtemp
		fmul	st,st(1)              ; s2
		fld	qword ptr [esi]       ; *iptr0
		fmul	st,st(5)              ; c2
		faddp	ST(1),ST
		fstp	qword ptr [esi]

;              rptr0 += full_span ;
;              iptr0 += full_span ;
		mov	EAX,-016Ch[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax
;              if (rptr0 < flagptr)
;                 continue ;
		cmp	edi,-04Ch[EBP]
		jb      short	L12CD

;              rtemp = s1 * s2
;              s2 = s1 * c2 + c1 * s2
;              c2 = c1 * c2 - rtemp
                fld     st(2)                    ; s1
		fmul	st,st(4)                 ; c2
		fld	st(2)                    ; c1
		fmul	st,st(2)                 ; s2
		faddp	ST(1),ST                 ; Sum (new s2) now in st(0)
                fld     st(3)                    ; s1
                fmulp   st(2),st                 ; s2 (rtemp now in st(1))
                                      ; up 1   rtemp in st(1)  new s2 in st(0)
		fld	st(2)                    ; c1
		fmulp	st(5),st                 ; c2
                fld     st(1)                    ; rtemp
		fsubp	st(5),st                 ; c2 = (c1*c2) - rtemp
                fstp    st(1)                    ; s2

;           if (rptr0 >= flagptr2)
;              break ;

		cmp	edi,-044h[EBP]
		jae     short	L1392

;              rptr0 += j ;
;              iptr0 += j ;
L137E:		mov	EAX,-0184h[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax
		jmp	near ptr L12CD  ; Back to top of inner loop

;           if (rptr0 >= flagptr2)
;              break ;

L1392:
		cmp	edi,-048h[EBP]
		jae     short	L1400    ; Break out of middle loop

;           rptr0 += l ;
;           iptr0 += l ;
		mov	EAX,-017Ch[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax

;  Update trig

;           c2 = c1 - (c0 * c1 + s0 * s1)
		fld	st(5)                    ; c0
		fmul	st,st(2)                 ; c1
		fld	st(5)                    ; s0
		fmul	st,st(4)                 ; s1
		faddp	ST(1),ST
		fld	st(2)                    ; c1
		fsubrp	ST(1),ST
                fstp    st(4)                    ; c2

;           s1 += (s0 * c1 - c0 * s1)
		fld	st(4)                    ; s0
		fmul	st,st(2)                 ; c1
		fld	st(6)                    ; c0
		fmul	st,st(4)                 ; s1
		fsubp	ST(1),ST
		fadd	st,st(3)                 ; s1
                fstp    st(3)                    ; s1

		jmp	near ptr L12A4  ; Back to top of middle loop

L1400:     ; Broke out of middle loop.  First, pop s1 and c2 to clean FPT stack.
                fstp    st(0)     ; s2
                fstp    st(0)     ; c1
                fstp    st(0)     ; s1
                fstp    st(0)     ; c2

;        if (rptr0 >= finalptr)
;           break ;
		cmp	edi,-040h[EBP]
		jb	L140D
;  All done with this odd kernel.  Clean up fpt stack, jump to kernel loop.
                finit
		jmp	near ptr LD4

;        rptr0 += m ;
;        iptr0 += m ;
L140D:		mov	EAX,-0178h[EBP]
		shl	EAX,3
                add     edi,eax
                add     esi,eax
		jmp	near ptr L1288  ; Back to top of outer loop

_TEXT	ends
_DATA	segment
_DATA	ends
_BSS	segment
_BSS	ends
	end
