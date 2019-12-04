#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=10  # memory in Mb
#SBATCH -o outfileSvmPos  # send stdout to outfile
#SBATCH -e errfileSvmPos  # send stderr to errfile
#SBATCH -t 0:30:00  # time requested in hour:minute:second

module --ignore-cache load Python/3.5.2-foss-2016b
module load pandas/0.19.0-foss-2016b-Python-3.5.2
python svm_pos.py ./project/data/train.csv
