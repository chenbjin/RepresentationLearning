#!/bin/bash

# Author:chenbingjin
# Date:2016-05-17
# Train word2vec
make
if [ ! -e train.txt ]; then
	printf "Train file 'train.txt' is acquired." 
else
	time ./word2vec -train train.txt -output vectors200.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 1 -iter 15
fi