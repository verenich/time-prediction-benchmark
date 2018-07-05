# Specifies an email address
#PBS -M ilyavere@gmail.com

# Says that an email will be sent to the specified address 
# at the beginning (-b) and end (-e) of a job and in case the job gets aborted (-a)
#PBS -m bea

# Sets the working directory of this jobscript
##PBS -d /home/n9334378/time-prediction-benchmark/experiments/

# Here, finally you can put the actual commands of your job, that will be run
# on the cluster.
source /etc/profile.d/modules.sh
module load python/3.6.4-intel-2017a
python -V
export PYTHONPATH=/home/n9334378/time-prediction-benchmark/
cd /home/n9334378/time-prediction-benchmark/experiments
echo "started hyperparameter optimization at $(date)"
python experiments_param_optim.py $dataset $method $encoding $learner
echo "finished hyperparameter optimization at $(date)"
python extract_best_params.py
echo "started experiments with optimal parameters at $(date)"
python experiments_final.py $dataset $method $encoding $learner
echo "finished experiments with optimal parameters at $(date)"
