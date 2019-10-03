#!/bin/bash

file_id="1"
number=1
counter=1
model="constrained"

while [ $counter -le 5 ]
do
	# python parser.py --algo ac --model_type mle --model_size $model --hidden_size 1 --env lin_dyn --max_actions 200 --states_dim 30 --salient_states_dim 2 --num_iters 1500 --real_episodes 1 --initial_model_lr 1e-4 --virtual_episodes 100 --num_action_repeats 1 --planning_horizon 10 --batch_size 128 --rhoER 0.25 --rs $file_id --file_id 'rho0.25_hidden1_criticfull_actorfull_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env lin_dyn --max_actions 200 --states_dim 30 --salient_states_dim 2 --num_action_repeats 1 --verbose 1 --save_checkpoints_training True --rs $file_id --file_id 'pi_salient_'$model'Model_'$file_id &

	python parser.py --algo ac --model_type paml --model_size $model --hidden_size 1 --env lin_dyn --max_actions 200 --states_dim 30 --salient_states_dim 2 --num_iters 200 --real_episodes 1 --initial_model_lr 1e-3 --virtual_episodes 100 --num_action_repeats 1 --planning_horizon 10 --batch_size 128 --rhoER 0.25 --rs $file_id --file_id 'rho0.25_hidden1_criticfull_actorfull_'$model'Model_'$file_id &
	
	# python parser.py --algo ac --model_type mle --model_size $model --hidden_size 1 --env Pendulum-v0 --max_actions 200 --num_iters 1000 --batch_size 250 --states_dim 120 --salient_states_dim 3 --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 --planning_horizon 20 --initial_model_lr 1e-4 --rhoER 0.25 --rs $file_id --file_id 'hidden1_criticfull_actorrestrict_rho0.25_real5_virt500_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env Pendulum-v0 --max_actions 200 --states_dim 120 --salient_states_dim 3 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size $model --hidden_size 1 --env Pendulum-v0 --max_actions 200 --states_dim 120 --salient_states_dim 3 --num_iters 200 --batch_size 250 --real_episodes 5 --virtual_episodes 500 --num_action_repeats 1 --planning_horizon 20 --initial_model_lr 1e-3 --rhoER 0.25 --rs $file_id --file_id 'hidden1_criticfull_actorfull_rho0.25_real5_virt500_'$model'Model_'$file_id &
	
	# python parser.py --algo ac --model_type mle --model_size $model --hidden_size 3 --env dm-Pendulum-v0 --max_actions 1000 --num_iters 1000 --states_dim 3 --salient_states_dim 3 --real_episodes 1 --virtual_episodes 80 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-4 --rhoER 0.5 --rs $file_id --file_id 'hidden3_rho0.5_virt80_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env dm-Cartpole-swingup-v0 --max_actions 1000 --states_dim 5 --salient_states_dim 5 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size $model --hidden_size 3 --env dm-Pendulum-v0 --max_actions 1000 --states_dim 3 --salient_states_dim 3 --num_iters 400 --real_episodes 1 --virtual_episodes 80 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --rs $file_id --file_id 'hidden3_rho0.5_virt80_'$model'Model_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running ac experiments... 

# srun -c 1 --mem=1G -p cpu