#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=10  # memory in Mb
#SBATCH -o outfileBayesTfidf  # send stdout to outfile
#SBATCH -e errfileBayesTfidf  # send stderr to errfile
#SBATCH -t 0:01:00  # time requested in hour:minute:second

module load Python/2.7.14-foss-2018a
module load pandas/0.19.1-intel-2016b-Python-2.7.12
python bayes_tfidf.py ./project/data/train.csv
