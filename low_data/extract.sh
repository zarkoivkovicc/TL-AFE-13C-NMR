#!/bin/bash
# call this script with "4033 3392" as the argument to reproduce paper results
samplings=$(ls -d */)
for sampling in $samplings
do
	cd $sampling
	python extract.py $@
	cd ..
done
