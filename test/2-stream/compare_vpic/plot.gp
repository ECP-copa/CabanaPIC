set term post eps enhanced 22 color

set output '2stream-minipic.eps'
set xlabel '{/Symbol w}_pt'
set ylabel 'W_E'
set xrange [0:60]
set yrange [1e-12:0.1]
set log y
set grid

f0(x) = exp(x)
f(x) = exp(0.497184855006572*2*x)
set keytitle '({/Symbol g}_0=1.0038), nx=32, nppc=4000'
set key bottom
wpe= 0.0195487
dv = 0.019746
plot 'out' u 2:($3*dv) w l t 'minipic 1-thread','energies' u ($1*wpe):2 w l t 'vpic 1-thread',f(x)*2e-12 t 'linear theory' 

