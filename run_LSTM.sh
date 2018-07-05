# Specifies an email address
#PBS -M ilyavere@gmail.com

# Says that an email will be sent to the specified address 
# at the beginning (-b) and end (-e) of a job and in case the job gets aborted (-a)
#PBS -m bea

# Sets the working directory of this jobscript
##PBS -d /home/n9334378/time-prediction-benchmark/

# Here, finally you can put the actual commands of your job, that will be run
# on the cluster.
source /etc/profile.d/modules.sh
module load tensorflow
python -V
cd /home/n9334378/time-prediction-benchmark/experiments
python train_LSTM.py $dataset $lstmsize 0.15 $nlayers $batchsize 0.001 $activation $optimizer
#python evaluate_LSTM.py $dataset $lstmsize 0.15 $nlayers $batchsize 0.001 $activation $optimizer
