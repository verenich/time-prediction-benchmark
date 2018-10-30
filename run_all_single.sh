#!/bin/bash -l
for LEARNER in svm
do
    for DATASET_NAME in bpic2012a hospital invoice bpic20151 bpic20152 bpic20153 bpic20154 bpic20155 bpic2017 traffic_fines production helpdesk bpic2012o bpic2012w sepsis bpic2011 #credit
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
                    memory=8gb
                fi
                qsub -l mem=$memory -l walltime=23:59:00 -l nodes=1:ppn=1 -N job2_"$DATASET_NAME"_"$BUCKET_METHOD"_"$CLS_ENCODING"_"$LEARNER" -v dataset=$DATASET_NAME,method=$BUCKET_METHOD,encoding=$CLS_ENCODING,learner=$LEARNER run.sh
            done
        done
    done
done
