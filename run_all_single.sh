#!/bin/bash -l
for LEARNER in xgb
do
    for DATASET_NAME in sepsis #bpic2011 bpic2015 bpic2017 traffic_fines
    #for DATASET_NAME in 
    do
        for BUCKET_METHOD in single prefix state cluster
        do
            for CLS_ENCODING in agg laststate index combined
            do
                if [ $DATASET_NAME == "sepsis" ] ; then
                    memory=10gb
                elif [ $DATASET_NAME == "traffic_fines" ] ; then
                    memory=15gb
                else
                    memory=15gb
                fi
                qsub -l mem=$memory -l walltime=23:00:00 -N job_"$DATASET_NAME"_"$BUCKET_METHOD"_"$CLS_ENCODING"_"$LEARNER" -v dataset=$DATASET_NAME,method=$BUCKET_METHOD,encoding=$CLS_ENCODING,learner=$LEARNER run.sh
            done
        done
    done
done
