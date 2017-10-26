#!/bin/bash -l
for LEARNER in rf
do
    for DATASET_NAME in bpic2017
    #for DATASET_NAME in bpic2017 traffic_fines
    do
        for BUCKET_METHOD in state single cluster prefix
        do
            for CLS_ENCODING in agg index laststate combined
            do
                if [ $DATASET_NAME == "sepsis" ] ; then
                    memory=10gb
                elif [ $DATASET_NAME == "traffic_fines" ] ; then
                    memory=15gb
                else
                    memory=15gb
                fi
                qsub -l mem=$memory -l walltime=43:00:00 -N job_"$DATASET_NAME"_"$BUCKET_METHOD"_"$CLS_ENCODING"_"$LEARNER" -v dataset=$DATASET_NAME,method=$BUCKET_METHOD,encoding=$CLS_ENCODING,learner=$LEARNER run.sh
            done
        done
    done
done
