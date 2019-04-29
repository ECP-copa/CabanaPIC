#!/bin/bash

echo "Positional Parameters"
echo '$0 = ' $0 # this file
echo '$1 = ' $1 # path to reader

reader=$1

# Find every .dat file, and pipe it to the read
#find -name '*.dat' -exec $reader {} \;
#find -name 'time.txt' -exec cat {} \;

input='*.dat'

array=()
while IFS=  read -r -d $'\0'; do
    array+=("$REPLY")
done < <(find . -name "${input}" -print0)

for i in "${array[@]}"
do
   echo "==== $i ===="
   $reader $i
   echo "==== END $i ===="
done
