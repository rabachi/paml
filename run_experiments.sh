#!/bin/bash

file_id="1"
number=1
counter=1

while [ $counter -le 5 ]
do
	python parser.py --algo reinforce --model_type mle --max_actions 50 --states_dim 10 --initial_model_lr 1e-7 --file_id $file_id &

	python parser.py --algo reinforce --model_type paml --max_actions 50 --states_dim 10 --initial_model_lr 1e-4 --file_id $file_id &

	((counter++))
	let file_id+=$number
done 
wait
echo Running all experiments... 

# srun -c 1 --mem=1G -p cpu