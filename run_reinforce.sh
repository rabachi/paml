#!/bin/bash

file_id="1"
number=1
counter=1

while [ $counter -le 1 ]
do
	python parser.py --algo reinforce --model_type mle --max_actions 50 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --small_model True --initial_model_lr 1e-4 --real_episodes 5 --num_iters 200 --save_checkpoints_training True --file_id 'smallModel_'$file_id &

	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 2 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 20 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 30 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &
	# python parser.py --algo reinforce --model_type model_free --max_actions 50 --states_dim 50 --salient_states_dim 2 --extra_dims_stable True --verbose 100 --file_id 'smallModel_'$file_id &

	python parser.py --algo reinforce --model_type paml --max_actions 50 --states_dim 10 --salient_states_dim 2 --extra_dims_stable True --small_model True --initial_model_lr 1e-4 --real_episodes 5 --num_iters 200  --save_checkpoints_training True --file_id 'smallModel_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running reinforce experiments... 

# srun -c 1 --mem=1G -p cpu