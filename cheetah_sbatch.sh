#!/bin/bash
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=mfch
#SBATCH --output=mfcht%j.log
#SBATCH --array=1-5

# (while true; do top -b -n 1 -u abachiro; sleep 10; done) & 

module load pytorch1.0-cuda9.0-python3.6
. /h/abachiro/mjpro200-py.env

ulimit -u 1000


############################### ANT #######################################################

# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 512 --env MBRLAnt-v0 --max_actions 100 --states_dim 111 --salient_states_dim 111 --num_iters 60 --real_episodes 5 --virtual_episodes 200 --batch_size 500 --num_action_repeats 1 --planning_horizon 5 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id 'learn10steps_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type random --model_size nn --hidden_size 512 --env MBRLAnt-v0 --max_actions 100 --states_dim 111 --salient_states_dim 111 --num_iters 60 --real_episodes 5 --virtual_episodes 200 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type mle --model_size nn --hidden_size 512 --env MBRLAnt-v0 --max_actions 100 --states_dim 111 --salient_states_dim 111 --num_iters 1000 --real_episodes 5 --virtual_episodes 200 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type model_free --env MBRLAnt-v0 --max_actions 100 --states_dim 111 --salient_states_dim 111 --num_action_repeats 1 --verbose 5 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID
###############################################################################################


############################### SWIMMER #######################################################

# python parser.py --algo ac --model_type mle --model_size nn --hidden_size 11 --env MBRLSwimmer-v0 --max_actions 1000 --states_dim 9 --salient_states_dim 9 --noise_type random --num_iters 1000 --real_episodes 1 --virtual_episodes 300 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type random --model_size nn --hidden_size 128 --env MBRLSwimmer-v0 --max_actions 1000 --states_dim 9 --salient_states_dim 9 --num_iters 1000 --real_episodes 1 --virtual_episodes 300 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 11 --env MBRLSwimmer-v0 --max_actions 1000 --states_dim 9 --salient_states_dim 9 --noise_type random --num_iters 60 --real_episodes 1 --virtual_episodes 300 --batch_size 500 --num_action_repeats 1 --planning_horizon 5 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id 'learn10steps_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type model_free --env MBRLSwimmer-v0 --max_actions 1000 --states_dim 40 --salient_states_dim 9 --noise_type random --num_action_repeats 1 --verbose 1 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID
#############################################################################################


############################### HOPPER #######################################################

# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 36 --env MBRLHopper-v0 --max_actions 100 --states_dim 11 --salient_states_dim 11 --num_iters 60 --real_episodes 1 --virtual_episodes 300 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type mle --model_size nn --hidden_size 36 --env MBRLHopper-v0 --max_actions 100 --states_dim 11 --salient_states_dim 11 --num_iters 1000 --real_episodes 1 --virtual_episodes 300 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type model_free --env MBRLHopper-v0 --max_actions 100 --states_dim 11 --salient_states_dim 11 --num_action_repeats 1 --verbose 1 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID
##################################################################################################


############################### HalfCheetah #######################################################
python parser.py --algo ac --model_type model_free --env HalfCheetah-v2 --max_actions 1000 --states_dim 17 \
                  --salient_states_dim 17 --num_action_repeats 1 --verbose 5 --rs $SLURM_ARRAY_TASK_ID \
                  --file_id $SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 512 --env HalfCheetah-v2 --max_actions 1000 --states_dim 17 --salient_states_dim 17 --num_iters 60 --real_episodes 5 --virtual_episodes 200 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID


# python parser.py --algo ac --model_type mle --model_size nn --env HalfCheetah-v2 --hidden_size 512 --max_actions 1000 --states_dim 17 --salient_states_dim 17 --num_iters 1000 --real_episodes 5 --virtual_episodes 200 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs $SLURM_ARRAY_TASK_ID --file_id $SLURM_ARRAY_TASK_ID
#################################################################################################



############################### Walker #######################################################

# parser.py --algo ac --model_type model_free --env Walker2d-v2 --max_actions 100 --states_dim 24 --salient_states_dim 24 --num_action_repeats 1 --verbose 1 --rs 0

# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 512 --env Walker2d-v2 --max_actions 200 --states_dim 17 --salient_states_dim 17 --num_iters 60 --real_episodes 5 --virtual_episodes 200 --batch_size 500 --num_action_repeats 1 --planning_horizon 2 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 20 --rs 0

#################################################################################################



############################### Pendulum #######################################################

# python parser.py --algo ac --model_type model_free --env Pendulum-v0 --max_actions 200 --states_dim 3 \
#                   --salient_states_dim 3 --num_action_repeats 1 --verbose 1 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'critic0_constrainedModel_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type paml --model_size constrained --hidden_size 2 --env Pendulum-v0 \
#                   --max_actions 200 --states_dim 6 --salient_states_dim 3 --noise_type redundant --num_iters 200 --batch_size 250 \
#                   --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 --planning_horizon 10 \
#                   --initial_model_lr 1e-3 --rhoER 0.25 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'lin_redundant_fid_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type paml --model_size constrained --hidden_size 2 --env Pendulum-v0 \
#                   --max_actions 200 --states_dim 6 --salient_states_dim 3 --noise_type redundant --num_iters 200 --batch_size 250 \
#                   --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 --planning_horizon 10 \
#                   --initial_model_lr 1e-3 --rhoER 0.25 --rs $SLURM_ARRAY_TASK_ID \
#                   --file_id 'lin_redundant_fid_'$SLURM_ARRAY_TASK_ID

# python parser.py --algo ac --model_type mle --model_size constrained --hidden_size 2 --env Pendulum-v0 \
#                             --max_actions 200 --num_iters 1000 --batch_size 250 --states_dim 33 \
#                             --salient_states_dim 3 --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 \
#                             --planning_horizon 10 --initial_model_lr 1e-4 --rhoER 0.25 --rs $SLURM_ARRAY_TASK_ID \
#                             --file_id 'fid_'$SLURM_ARRAY_TASK_ID
#################################################################################################
echo Running ac experiments...