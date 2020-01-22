#!/bin/bash
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=mle_5_512_cheetah
#SBATCH --output=mle_5_512_cheetah_%j.log
#SBATCH --array=1-10

# (while true; do top -b -n 1 -u abachiro; sleep 10; done) & 

module load pytorch1.0-cuda9.0-python3.6
. /h/abachiro/mjpro200-py.env

ulimit -u 1000

# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 512 --env HalfCheetah-v2 --max_actions 1000 \
#                   --states_dim 17 --salient_states_dim 17 --num_iters 200 --real_episodes 1 --virtual_episodes 10 \
#                   --batch_size 500 --num_action_repeats 1 --planning_horizon 5 --initial_model_lr 1e-3 --rhoER 0.5 \
#                   --verbose 5 --rs $SLURM_ARRAY_TASK_ID --file_id 'fid_'$SLURM_ARRAY_TASK_ID
                  # >'output_cheetah0critic0_notadded_'$SLURM_ARRAY_TASK_ID'.txt'



# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env dm-Cartpole-balance-v0 \
#                 --max_actions 200 --states_dim 35 --salient_states_dim 5 --num_iters 200 --real_episodes 1 \
#                 --virtual_episodes 20 --batch_size 250 --num_action_repeats 1 --planning_horizon 3 \
#                 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $SLURM_ARRAY_TASK_ID \
#                 --file_id 'critic0_nnModel_'$SLURM_ARRAY_TASK_ID


# python parser.py --algo ac --model_type paml --model_size constrained --hidden_size 2 --env Pendulum-v0 \
#                   --max_actions 200 --states_dim 33 --salient_states_dim 3 --num_iters 200 --batch_size 250 \
#                   --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 --planning_horizon 10 \
#                   --initial_model_lr 1e-3 --rhoER 0.25 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'fid_'$SLURM_ARRAY_TASK_ID

python parser.py --algo ac --model_type mle --model_size nn --env HalfCheetah-v2 --hidden_size 512 --max_actions 1000 \
                  --states_dim 17 --salient_states_dim 17 --num_iters 2000 --real_episodes 1 --virtual_episodes 10 \
                  --batch_size 250 --num_action_repeats 1 --planning_horizon 5 --initial_model_lr 1e-5 --rhoER 0.5 \
                  --verbose 5 --rs $SLURM_ARRAY_TASK_ID --file_id 'fid_'$SLURM_ARRAY_TASK_ID


# python parser.py --algo ac --model_type mle --model_size constrained --hidden_size 2 --env Pendulum-v0 \
#                             --max_actions 200 --num_iters 1000 --batch_size 250 --states_dim 33 \
#                             --salient_states_dim 3 --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 \
#                             --planning_horizon 10 --initial_model_lr 1e-4 --rhoER 0.25 --rs $SLURM_ARRAY_TASK_ID \
#                             --file_id 'fid_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type mle --ensemble 1 --model_size nn --hidden_size 3 --env dm-Cartpole-balance-v0 \
#                   --max_actions 200 --num_iters 2000 --states_dim 35 --salient_states_dim 5 --real_episodes 1 \
#                   --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 \
#                   --rhoER 0.5 --batch_size 250 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'critic0_nnModel_'$SLURM_ARRAY_TASK_ID


# python parser.py --algo ac --model_type model_free --env HalfCheetah-v2 --max_actions 1000 --states_dim 17 \
#                   --salient_states_dim 17 --num_action_repeats 1 --verbose 1 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'fid_hidden512_'$SLURM_ARRAY_TASK_ID


# python parser.py --algo ac --model_type model_free --env Pendulum-v0 --max_actions 200 --states_dim 3 \
#                   --salient_states_dim 3 --num_action_repeats 1 --verbose 1 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'critic0_constrainedModel_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type model_free --env dm-Cartpole-balance-v0 --max_actions 200 --states_dim 20 \
#                   --salient_states_dim 5 --num_action_repeats 1 --verbose 1 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'final_critic10_nnModel_'$SLURM_ARRAY_TASK_ID

echo Running ac experiments...