#!/bin/bash
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=test_romina
#SBATCH --output=test_romina_%j.log
#SBATCH --array=1-10

# (while true; do top -b -n 1 -u abachiro; sleep 10; done) & 

module load pytorch1.0-cuda9.0-python3.6
. /h/abachiro/mjpro200-py.env

srun -c 1 --mem=4G python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env HalfCheetah-v2 --max_actions 1000 --states_dim 27 --salient_states_dim 17 --num_iters 200 --real_episodes 1 --virtual_episodes 20 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $SLURM_ARRAY_TASK_ID --file_id 'final_critic1extraep1000steps_notadded_nnModel_'$SLURM_ARRAY_TASK_ID #>'output_cheetah0critic0_notadded_'$SLURM_ARRAY_TASK_ID'.txt'


# srun -c 1 --mem=4G python parser.py --algo ac --model_type paml --model_size constrained --hidden_size 2 --env Pendulum-v0 --max_actions 200 --states_dim 3 --salient_states_dim 3 --num_iters 200 --batch_size 250 --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 --planning_horizon 5 --initial_model_lr 1e-3 --rhoER 0.25 --rs $SLURM_ARRAY_TASK_ID --file_id 'final_hidden2_rho0.25_real5_virt500_critic0extraeps_constrainedModel_'$SLURM_ARRAY_TASK_ID

echo Running ac experiments...



# #! /bin/bash
# #SBATCH -p cpu
# # # SBATCH --ntasks=30
# # # SBATCH --cpus-per-task=2
# # # SBATCH --mem-per-cpu=4G
# #SBATCH --job-name=test_romina
# #SBATCH --output=test_romina_%j.log

# module load pytorch1.0-cuda9.0-python3.6
# . /h/abachiro/mjpro200-py.env

# for i in {1..15}
# do
#    srun -c 2 --mem=8G -N1 -n1 -c1 --exclusive python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env HalfCheetah-v2 --max_actions 1000 --states_dim 27 --salient_states_dim 17 --num_iters 200 --real_episodes 1 --virtual_episodes 20 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $i --file_id 'final_critic10extraep1000steps_notadded_nnModel_'$i >'output_cheetahcritic10_notadded_'$i'.txt' &
# done
# wait


