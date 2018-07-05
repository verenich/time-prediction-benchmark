#!/bin/bash -l
for DATASET_NAME in production #bpic20151 bpic20152 bpic20153 bpic20154 bpic20155 helpdesk hospital minit sepsis bpic2011 bpic2017 traffic_fines
do
    for LSTM_SIZE in 50 100 150  #100 is fine
    do
        for N_LAYERS in 1 2 3  #2 is fine
        do
            for BATCH_SIZE in 8 16 32  #8 performs badly
            do
                for ACTIVATION in relu linear
                do
                    for OPTIMIZER in rmsprop adam #is worse, at least for remtime
                    do
                        if [ $DATASET_NAME == "sepsis" ] ; then
                            memory=3gb
                        elif [ $DATASET_NAME == "bpi15" ] ; then
                            memory=7gb
                        else
                            memory=4gb
                        fi
                        qsub -l mem=$memory -l walltime=23:59:00 -l nodes=1:ppn=5 -N lstm_"$DATASET_NAME"_"$LSTM_SIZE"_"$N_LAYERS"_"$BATCH_SIZE"_"$ACTIVATION"_"$OPTIMIZER" -v dataset=$DATASET_NAME,lstmsize=$LSTM_SIZE,nlayers=$N_LAYERS,batchsize=$BATCH_SIZE,activation=$ACTIVATION,optimizer=$OPTIMIZER run_LSTM.sh
                    done
                done
            done
        done
    done
done
