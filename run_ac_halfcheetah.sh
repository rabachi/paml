#!/bin/bash

file_id="1"
number=1
counter=1
model="nn"

while [ $counter -le 5 ]
do
	python parser.py --algo ac --model_type mle --model_size $model --hidden_size 17 --env HalfCheetah-v2 --max_actions 100 --num_iters 1000 --states_dim 17 --salient_states_dim 17 --real_episodes 1 --virtual_episodes 100 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --batch_size 512 --rs $file_id --file_id 'rho0.5_virt100_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size $model --hidden_size 17 --env HalfCheetah-v2 --max_actions 200 --states_dim 17 --salient_states_dim 17 --num_iters 600 --real_episodes 1 --virtual_episodes 110 --num_action_repeats 1 --planning_horizon 5 --initial_model_lr 1e-3 --rhoER 0.5 --batch_size 128 --rs $file_id --file_id 'rho0.5_virt110_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env HalfCheetah-v2 --max_actions 200 --states_dim 17 --salient_states_dim 17 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running ac experiments... 

# srun -c 1 --mem=1G -p cpu