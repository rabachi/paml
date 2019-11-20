#!/bin/bash
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=test_romina
#SBATCH --output=test_romina_%j.log
#SBATCH --array=19-21

module load pytorch1.0-cuda9.0-python3.6
. /h/abachiro/mjpro200-py.env

ulimit -u 1000

echo Running reinforce experiments... reinforce_paml_states10_traj200 12-21

# srun -c 2 --mem=8G 
python parser.py --algo reinforce --model_type paml --max_actions 200 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --model_size small --initial_model_lr 1e-7 --real_episodes 5 --num_iters 200 --virtual_episodes 1000 --rs $SLURM_ARRAY_TASK_ID --save_checkpoints_training False --file_id 'final_smallModel_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo reinforce --model_type mle --max_actions 200 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --model_size small --initial_model_lr 1e-8 --real_episodes 5 --num_iters 200 --virtual_episodes 1000 --rs $SLURM_ARRAY_TASK_ID --save_checkpoints_training False --file_id 'smallModel_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo reinforce --model_type model_free --max_actions 200 --states_dim 2 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --rs $SLURM_ARRAY_TASK_ID --file_id 'final_smallModel_'$SLURM_ARRAY_TASK_ID
