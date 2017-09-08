#!/bin/bash -l

#for DATASET_NAME in bpic2015 sepsis bpic2011 bpic2017 traffic_fines
for DATASET_NAME in sepsis
do
    for BUCKET_METHOD in prefix single
    do
        for CLS_ENCODING in agg index laststate combined
        do
            if [ $DATASET_NAME == "sepsis" ] ; then
                memory=15gb
            elif [ $DATASET_NAME == "traffic_fines" ] ; then
                memory=15gb
            else
                memory=15gb
            fi
            qsub -l mem=$memory -l walltime=46:00:00 -N job_$DATASET_NAME$BUCKET_METHOD$CLS_ENCODING -v dataset=$DATASET_NAME,method=$BUCKET_METHOD,encoding=$CLS_ENCODING run.sh
        done
    done
done
