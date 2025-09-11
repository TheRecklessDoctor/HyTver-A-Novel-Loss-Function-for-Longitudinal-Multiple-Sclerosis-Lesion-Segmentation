#!/bin/env bash

#subdir=111
# for MSSEG'2
#       $1 is the name of the original data folder.
#       $2 is the name of the folder which to save the skull-stripped data.
for dir in ../"$1"/*
do
    subdir=$(basename "$dir")
    outDir="./$2/$subdir/"
    mkdir -p $outDir
    echo "Processing $dir -> $outDir"
    hd-bet -i "$dir" -o "$outDir" --save_bet_mask 
    #((subdir+=1))
done

