#!/bin/bash

file_id="1"
number=1
counter=1

while [ $counter -le 5 ]
do
	# python parser.py --algo ac --model_type mle --small_model True --env lin_dyn --max_actions 200 --states_dim 10 --salient_states_dim 2 --num_iters 200 --real_episodes 5 --virtual_episodes 100 --num_action_repeats 1 --file_id 'smallModel_searchCtrl_'$file_id &

	python parser.py --algo ac --model_type model_free --env lin_dyn --max_actions 200 --states_dim 10 --salient_states_dim 2 --num_action_repeats 2 --file_id 'smallModel_searchCtrl_repeats2_'$file_id &

	# python parser.py --algo ac --model_type paml --small_model True --env lin_dyn --max_actions 200 --states_dim 10 --salient_states_dim 2 --num_iters 200 --real_episodes 5 --virtual_episodes 100 --num_action_repeats 1 --file_id 'smallModel_searchCtrl_'$file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running ac experiments... 

# srun -c 1 --mem=1G -p cpu