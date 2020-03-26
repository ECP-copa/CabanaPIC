set term png enhanced 

set output '2pcle-minipic.png'
set xlabel 't'
set ylabel 'particle location'
set xrange [0:100]
set yrange [0:1]
set grid

f(x)=0.7-0.25-0.5*(0.2-0.5)*cos(x)
set keytitle 'nx=1000'
set key bottom

plot 'partloc' u 1:2 w l t 'minipic','partloc-vpic' u 1:2 w l t 'vpic', f(x) t 'theory'
