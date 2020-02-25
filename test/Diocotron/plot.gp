set term png enhanced 

set output 'diocotron-comp.png'
set xlabel '{/Symbol w}_pt'
set ylabel '|E_x|(m=1)'
set xrange [0:350]
set yrange [2e-3:5]
set log y
set grid

f(x) = exp(0.2/sqrt(20)*x)
set key bottom

plot '/home/scratch/guangyechen/vpic-master/vpic/test/Diocotron/plot/ex-fft.txt' u 1:3 w l t 'vpic','/home/scratch/guangyechen/vpic-master/pcle-ghost/ivpic-parallel-comms/test/Diocotron/plot/ex-fft.txt' u 1:3 w l t 'ivpic','ex-fft.txt' u 1:3 w l t 'cabanapic',f(x)*0.012 t 'linear theory'