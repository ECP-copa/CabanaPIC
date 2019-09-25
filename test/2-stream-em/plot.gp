set term png #post eps enhanced 22 color

#set output '2stream-em.eps'
set output '2stream-em.png'

set xlabel '{/Symbol w}_pt'
set ylabel 'W_B'
set xrange [0:150]
set yrange [1e-14:10]
set log y
set grid
set format y "%.e"
f(x) = exp(0.279*2*x)
set keytitle '({/Symbol g}_0=1.02), nx=32, nppc=100'
set key bottom

plot 'outw0.2' u 2:4 w l t 'minipic 1-thread',f(x)*5e-15 t 'linear theory' 

