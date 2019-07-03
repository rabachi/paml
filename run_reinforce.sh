#!/bin/bash

file_id="1"
number=1
counter=1

while [ $counter -le 5 ]
do
	python parser.py --algo reinforce --model_type mle --max_actions 200 --states_dim 30 --salient_states_dim 2 --extra_dims_stable True --model_size small --initial_model_lr 1e-8 --real_episodes 5 --num_iters 200 --virtual_episodes 1000 --rs $file_id --save_checkpoints_training False --file_id 'smallModel_'$file_id &

	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --rs $file_id --file_id 'smallModel_'$file_id &

	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 20 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 30 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 50 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &

	python parser.py --algo reinforce --model_type paml --max_actions 200 --states_dim 30 --salient_states_dim 2 --extra_dims_stable True --model_size small --initial_model_lr 1e-7 --real_episodes 5 --num_iters 200 --virtual_episodes 1000 --rs $file_id --save_checkpoints_training False --file_id 'smallModel_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running reinforce experiments... 

# srun -c 1 --mem=1G -p cpu