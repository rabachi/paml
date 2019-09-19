#!/bin/bash

file_id="1"
number=1
counter=1
model="nn"

while [ $counter -le 5 ]
do
	# python parser.py --algo ac --model_type mle --ensemble 1 --model_size $model --hidden_size 3 --env Reacher-v2 --max_actions 100 --num_iters 2000 --states_dim 11 --salient_states_dim 11 --real_episodes 1 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-4 --rhoER 0.5 --rs $file_id --file_id 'traineverytraj_lrsched_virt20_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env Reacher-v2 --max_actions 10 --states_dim 11 --salient_states_dim 11 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type random --ensemble 1 --model_size $model --hidden_size 3 --env Reacher-v2 --max_actions 10 --states_dim 501 --salient_states_dim 11 --num_iters 1000 --batch_size 500 --real_episodes 1 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --rs $file_id --file_id $model'Model_'$file_id &

	python parser.py --algo ac --model_type paml --ensemble 1 --model_size $model --hidden_size 3 --env Reacher-v2 --max_actions 100 --states_dim 501 --salient_states_dim 11 --num_iters 400 --batch_size 500 --real_episodes 2 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --rs $file_id --file_id $model'Model_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running ac experiments... 

# srun -c 1 --mem=1G -p cpu