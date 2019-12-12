set term png #post eps enhanced 22 color

#set output '2stream-em.eps'
set output '2stream-em.png'

set xlabel '{/Symbol w}_pt'
set ylabel 'W_B'
set xrange [0:150]
set yrange [1e-15:10]
set log y
set grid
set format y "%.e"
f(x) = exp(0.279*2*x)
set keytitle '({/Symbol g}_0=1.02), nx=32, nppc=100'
set key bottom

plot 'outw0.2' u 2:4 w l t 'minipic 1-thread','outw0.2-2' u 2:4 w l t 'minipic 2-thread',f(x)*1e-16 t 'linear theory' 


set output '2stream-em-comp.png'
dV = 0.019746
dt = 0.0195487
set xrange [0:120]
set yrange [1e-15:1e-4]
set keytitle 'nx=32, nppc=50'
plot 'energies-em0' u ($1*dt):7 w l t 'vpic','out-em1' u 2:($4*dV) w l t 'minipic'
