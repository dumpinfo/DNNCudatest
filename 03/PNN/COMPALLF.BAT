set ipath=c:\sc\include
del temp
del *.obj
c:\sc\bin\sc basic -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc brentmin  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc confuse  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc control  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc defaults  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc dermin  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc dotprod  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc execute  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc gcnn  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc glob_min  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc mem  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc messages  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc parsdubl  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc pnn  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc pnnet  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc process  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc prog_win  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc sepclass  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc sepvar  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc test  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc train  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc veclen  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp
c:\sc\bin\sc wt_save  -a4 -A -3 -bx -c -ff -mx -r -v1 >>temp

