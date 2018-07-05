#!/bin/bash -l
for LEARNER in xgb
do
    for DATASET_NAME in production #credit bpic2012o bpic2012w credit helpdesk hospital invoice sepsis bpic2011 bpic2015 bpic2017 traffic_fines
    #for DATASET_NAME in 
    do
        for BUCKET_METHOD in cluster single prefix state
        do
            for CLS_ENCODING in agg laststate index
            do
                if [ $DATASET_NAME == "hospital" ] ; then
                    memory=15gb
                elif [ $DATASET_NAME == "traffic_fines" ] ; then
                    memory=15gb
                else
                    memory=4gb
                fi
                qsub -l mem=$memory -l walltime=16:00:00 -l nodes=1:ppn=2 -N job_"$DATASET_NAME"_"$BUCKET_METHOD"_"$CLS_ENCODING"_"$LEARNER" -v dataset=$DATASET_NAME,method=$BUCKET_METHOD,encoding=$CLS_ENCODING,learner=$LEARNER run.sh
            done
        done
    done
done
