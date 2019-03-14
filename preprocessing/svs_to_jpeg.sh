#!/bin/bash

for filename in $1/*.svs; do python ~/DeepPATH/DeepPATH_code/00_preprocessing/0b_tileLoop_deepzoom4.py -s 256 -e 0 -j 64 -o $2/${filename##*/} $filename; done
