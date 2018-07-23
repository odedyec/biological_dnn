#!/usr/bin/env bash

 
for ((i=1;i<=123;i++)); 
    do
	arr=()
	echo "======================"
	filename="train/TF${i}_pbm.txt"
	arr+=($filename)
	for ((j=0;j<=6;j++));
	    do
		filename="train/TF${i}_selex_${j}.txt"
		if [ -e $filename ]
		then
		    arr+=($filename)
		fi
	done;
	python main.py ${arr[@]}
    done;


