set term post eps enhanced 22 color

set output '2stream-minipic.eps'
set xlabel '{/Symbol w}_pt'
set ylabel 'W_E'
set xrange [0:50]
set yrange [1e-8:100]
set log y
set grid

f0(x) = exp(x)
f(x) = exp(0.497184855006572*2*x)
set keytitle '({/Symbol g}_0=1.0038), nx=32, nppc=8000'
set key bottom

plot 'out' u 2:3 w l t 'minipic 1-thread','out2' u 2:3 w l t 'minipic 2-thread',f(x)*1.2e-10 t 'linear theory' #,f0(x)*1.2e-10

