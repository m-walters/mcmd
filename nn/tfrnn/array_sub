#!/bin/bash
dataname="xmelt"
datapath="/scratch/walterms/mcmd/output/xmelt/processed/"
outpath="/scratch/walterms/mcmd/output/xmelt/sbatchlogs/"
outfname="xmelt"
#sbatch --output=$outpath"logs/"$outfname"_%a.log" --array=0-5 sbatch_array_run $datapath$dataname $outpath$outfname
sbatch --output=$outpath$outfname"_%a.log" --array=0-5 train $datapath$dataname $outpath$outfname
