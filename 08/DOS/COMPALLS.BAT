set ipath=c:\sc\include
del temp
del *.obj
c:\sc\bin\sc display  -a4 -3 -bx -c -ff -mx -r -s -v1 >>temp
c:\sc\bin\sc fg_cstm  -a4 -3 -bx -c -ff -mx -r -s -v1 >>temp
c:\sc\bin\sc graphics  -a4 -3 -bx -c -ff -mx -r -s -v1 >>temp
c:\sc\bin\sc npredict  -a4 -A -3 -bx -c -ff -mx -r -s -v1 >>temp
c:\sc\bin\sc prog_win  -a4 -A -3 -bx -c -ff -mx -r -s -v1 >>temp

