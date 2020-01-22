#!/bin/bash
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=reinforce_paml_2_traj200
#SBATCH --output=reinforce_paml_2_traj200_%j.log
#SBATCH --array=1-10

module load pytorch1.0-cuda9.0-python3.6
. /h/abachiro/mjpro200-py.env

ulimit -u 1000

echo Running reinforce experiments... reinforce_paml_states2_traj200

# srun -c 2 --mem=8G 
 python parser.py --algo reinforce --model_type paml --max_actions 200 --states_dim 2 --salient_states_dim 2 \
        --extra_dims_stable True --model_size small --initial_model_lr 1e-4 --real_episodes 5 --num_iters 400 \
        --virtual_episodes 500 --rs $SLURM_ARRAY_TASK_ID --save_checkpoints_training True \
        --file_id 'fid4_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo reinforce --model_type mle --max_actions 200 --states_dim 2 --salient_states_dim 2 \
#      --extra_dims_stable True --model_size small --initial_model_lr 1e-5 --real_episodes 5 --num_iters 400 \
#      --virtual_episodes 2000 --rs $SLURM_ARRAY_TASK_ID --save_checkpoints_training True \
#      --file_id 'fid_'$SLURM_ARRAY_TASK_ID

#python parser.py --algo reinforce --model_type model_free --max_actions 200 --states_dim 50 --salient_states_dim 2 \
#      --extra_dims_stable True --verbose 100 --rs $SLURM_ARRAY_TASK_ID --file_id 'fid_'$SLURM_ARRAY_TASK_ID
