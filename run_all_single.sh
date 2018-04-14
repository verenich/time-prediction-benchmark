#!/bin/bash -l
for DATASET_NAME in CR
do
    for LSTM_SIZE in 100 200  #100 is fine
    do
        for N_LAYERS in 1 2 3  #2 is fine
        do
            for BATCH_SIZE in 32  #8 performs badly
            do
                for ACTIVATION in relu linear
                do
                    for OPTIMIZER in rmsprop #adam is worse, at least for remtime
                    do
                        if [ $DATASET_NAME == "sepsis" ] ; then
                            memory=3gb
                        elif [ $DATASET_NAME == "bpi15" ] ; then
                            memory=7gb
                        else
                            memory=4gb
                        fi
                        qsub -l mem=$memory -l walltime=12:00:00 -l nodes=1:ppn=5 -N job_"$DATASET_NAME"_"$LSTM_SIZE"_"$N_LAYERS"_"$BATCH_SIZE"_"$ACTIVATION"_"$OPTIMIZER" -v dataset=$DATASET_NAME,lstmsize=$LSTM_SIZE,nlayers=$N_LAYERS,batchsize=$BATCH_SIZE,activation=$ACTIVATION,optimizer=$OPTIMIZER run.sh
                    done
                done
            done
        done
    done
done
