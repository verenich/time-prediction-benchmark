#!/bin/bash -l

#for DATASET_NAME in bpic2015 sepsis bpic2011 bpic2017 traffic_fines
export PYTHONPATH=/home/n9334378/time-prediction-benchmark/
#module load python
cd experiments
for DATASET_NAME in sepsis
do
    for BUCKET_METHOD in prefix single
    do
        for CLS_ENCODING in agg index laststate combined
        do
            echo "starting $DATASET_NAME $BUCKET_METHOD $CLS_ENCODING at $(date)"
            python experiments_param_optim.py $DATASET_NAME agg $BUCKET_METHOD $CLS_ENCODING
        done
    done
done