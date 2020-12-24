set ipath=c:\sc\include
del temp
del *.obj
c:\sc\bin\sc activity -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc act_func -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc anneal1 -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc anneal2 -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc an1 -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc an2 -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc anx_dd -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc confuse -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc conjgrad -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc control -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc direcmin -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc dotprod -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc dotprodc -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc execute -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc flrand -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc gradient -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc layernet -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc lev_marq -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc limit -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc lm_core -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc mem -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc messages -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc mlfn2 -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc parsdubl -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc perturb -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc random -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc regress -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc regrs_dd -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc shake -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc ssg -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc svdcmp -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc test -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc train -a4 -A -3 -bx -c -mx -r -v1 >>temp
c:\sc\bin\sc wt_save -a4 -A -3 -bx -c -mx -r -v1 >>temp

