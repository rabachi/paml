#!/bin/bash

file_id="1"
number=1
counter=1
model="nn"

while [ $counter -le 10 ]
do
	# python parser.py --algo ac --model_type mle --ensemble 1 --model_size $model --hidden_size 3 --env dm-Walker-v0 --max_actions 1000 --num_iters 2000 --states_dim 29 --salient_states_dim 24 --real_episodes 1 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --batch_size 250 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --ensemble 1 --model_size $model --hidden_size 3 --env dm-Walker-v0 --max_actions 500 --states_dim 24 --salient_states_dim 24 --num_iters 100 --real_episodes 1 --virtual_episodes 50 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --batch_size 500 --rs $file_id --file_id 'trainmodelevery_every250_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env dm-Walker-v0 --max_actions 1000 --states_dim 24 --salient_states_dim 24 --num_iters 1000 --real_episodes 5 --virtual_episodes 100 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-4 --rhoER 0.0 --verbose 5 --rs $file_id --file_id $model'Model_'$file_id &
	# python parser.py --algo ac --model_type model_free --env dm-Walker-v0 --max_actions 1000 --states_dim 29 --salient_states_dim 24 --num_action_repeats 1 --batch_size 200 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type model_free --env HalfCheetah-v2 --max_actions 1000 --states_dim 27 --salient_states_dim 17 --num_action_repeats 1 --batch_size 200 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &
	

	# python parser.py --algo ac --model_type mle --ensemble 1 --model_size nn --hidden_size 3 --env dm-Cartpole-balance-v0 --max_actions 200 --num_iters 2000 --states_dim 20 --salient_states_dim 5 --real_episodes 1 --virtual_episodes 20 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --batch_size 250 --rs $file_id --file_id 'final_'$model'Model_'$file_id > 'errors_mle_cartpole.txt' &

	# python parser.py --algo ac --model_type model_free --env dm-Cartpole-balance-v0 --max_actions 200 --states_dim 20 --salient_states_dim 5 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &


	
	# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env dm-Cartpole-balance-v0 --max_actions 200 --states_dim 5 --salient_states_dim 5 --num_iters 200 --real_episodes 1 --virtual_episodes 20 --batch_size 250 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $file_id --file_id 'final_critic0extraep1000steps_'$model'Model_'$file_id &





	# python parser.py --algo ac --model_type model_free --env HalfCheetah-v2 --max_actions 1000 --states_dim 22 --salient_states_dim 17 --num_action_repeats 1 --verbose 1 --rs $file_id --file_id $model'Model_'$file_id &





	/h/abachiro/.local/bin/mprof run -o 'maxdataset_cheetah27_critic0_'$file_id'.dat' python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env HalfCheetah-v2 --max_actions 1000 --states_dim 27 --salient_states_dim 17 --num_iters 200 --real_episodes 1 --virtual_episodes 20 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $file_id --file_id 'maxdataset_final_critic0extraep1000steps_'$model'Model_'$file_id &





	# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env dm-Walker-v0 --max_actions 1000 --states_dim 29 --salient_states_dim 24 --num_iters 1000 --real_episodes 1 --virtual_episodes 20 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $file_id --file_id $model'Model_'$file_id &

	# python parser.py --algo ac --model_type random --model_size nn --hidden_size 3 --env HalfCheetah-v2 --max_actions 1000 --states_dim 17 --salient_states_dim 17 --num_iters 600 --real_episodes 1 --virtual_episodes 20 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-3 --rhoER 0.5 --verbose 5 --rs $file_id --file_id 'exprad0.8_'$model'Model_'$file_id &
# 
	# python parser.py --algo ac --model_type mle --model_size nn --hidden_size 3 --env HalfCheetah-v2 --max_actions 1000 --states_dim 17 --salient_states_dim 17 --num_iters 2000 --real_episodes 1 --virtual_episodes 20 --batch_size 250 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-5 --rhoER 0.5 --verbose 5 --rs $file_id --file_id 'final_'$model'Model_'$file_id &

	# python parser.py --algo ac --model_type paml --model_size nn --hidden_size 3 --env dm-Walker-v0 --max_actions 1000 --states_dim 24 --salient_states_dim 24 --num_iters 1000 --real_episodes 5 --virtual_episodes 100 --batch_size 500 --num_action_repeats 1 --planning_horizon 3 --initial_model_lr 1e-4 --rhoER 0.0 --verbose 5 --rs $file_id --file_id $model'Model_'$file_id &
	
	((counter++))
	let file_id+=$number
done 
wait
echo Running ac experiments... 

# srun -c 1 --mem=1G -p cpu