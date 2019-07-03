#!/bin/bash

file_id="1"
number=1
counter=1
model="small"

while [ $counter -le 5 ]
do
	# python parser.py --algo ac --model_type mle --model_size $model --hidden_size 3 --env lin_dyn --max_actions 200 --states_dim 30 --salient_states_dim 2 --num_iters 200 --real_episodes 1 --initial_model_lr 1e-4 --virtual_episodes 10 --num_action_repeats 1 --planning_horizon 8 --rs $file_id --file_id 'noiseless_planning_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env lin_dyn --max_actions 200 --states_dim 30 --salient_states_dim 2 --num_action_repeats 1 --verbose 1 --save_checkpoints_training True --rs $file_id --file_id 'noiseless_planning_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size $model --hidden_size 3 --env lin_dyn --max_actions 200 --states_dim 30 --salient_states_dim 2 --num_iters 200 --real_episodes 1 --initial_model_lr 1e-4 --virtual_episodes 10 --num_action_repeats 1 --planning_horizon 8 --rs $file_id --file_id 'noiseless_planning_'$model'Model_'$file_id &
	
	# python parser.py --algo ac --model_type mle --model_size $model --hidden_size 3 --env Pendulum-v0 --max_actions 200 --num_iters 200 --states_dim 30 --salient_states_dim 3 --real_episodes 1 --virtual_episodes 100 --num_action_repeats 1 --planning_horizon 8 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env Pendulum-v0 --max_actions 200 --states_dim 30 --salient_states_dim 3 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size $model --hidden_size 3 --env Pendulum-v0 --max_actions 200 --states_dim 30 --salient_states_dim 3 --num_iters 100 --real_episodes 1 --virtual_episodes 100 --num_action_repeats 1 --planning_horizon 8 --rs $file_id --file_id $model'Model_'$file_id &
	
	# python parser.py --algo ac --model_type mle --model_size $model --hidden_size 3 --env dm-Pendulum-v0 --max_actions 1000 --num_iters 200 --states_dim 3 --salient_states_dim 3 --real_episodes 5 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 8 --rs $file_id --file_id $model'Model_'$file_id &

	python parser.py --algo ac --model_type model_free --env dm-Pendulum-v0 --max_actions 1000 --states_dim 3 --salient_states_dim 3 --num_action_repeats 1 --verbose 5 --rs $file_id --file_id 'updates_after_ep_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size $model --hidden_size 3 --env dm-Pendulum-v0 --max_actions 1000 --states_dim 3 --salient_states_dim 3 --num_iters 100 --real_episodes 5 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 8 --rs $file_id --file_id $model'Model_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running ac experiments... 

# srun -c 1 --mem=1G -p cpu