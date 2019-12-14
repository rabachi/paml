# PAML

### TODO:
* Link PAML paper here

# Running Experiments

## Setup Environment

Create a Python 3.7 environment and install required packages:
```bash
conda create -n py37 python=3.7
source activate py37
pip install -r requirements.txt
```
This project also requires mujoco 2.0, which can be downloaded from: https://mujoco.org/

## Experiments

Sample commands for running experiments:

### REINFORCE (Williams 1992)

To run PAML experiment e.g. for states_dim = 50 (48 irrelevant dims) (see additional options by using help):
```
python parser.py --algo reinforce --model_type paml --max_actions 200 --states_dim 50 --salient_states_dim 2 \
        --extra_dims_stable True --model_size small --initial_model_lr 1e-4 --real_episodes 5 --num_iters 250 \
        --virtual_episodes 2000 --rs 1 --save_checkpoints_training False --file_id 'fid_1'
```
To plot above experiment: 
```
python plotting.py --algo reinforce --model_type paml --states_dim 50 --salient_states_dim 2 \
                     --max_actions 200 --file_id 'fid' --num_files 1 --model_size small
```

To run MLE:
```
python parser.py --algo reinforce --model_type mle --max_actions 200 --states_dim 50 --salient_states_dim 2 \
     --extra_dims_stable True --model_size small --initial_model_lr 1e-5 --real_episodes 5 --num_iters 400 \
     --virtual_episodes 2000 --rs $SLURM_ARRAY_TASK_ID --save_checkpoints_training True \
     --file_id 'fid_'$SLURM_ARRAY_TASK_ID
```
To plot both experiments:
```
python plotting.py --algo reinforce --model_type paml,mle --states_dim 50 --salient_states_dim 2 \
                     --max_actions 200 --file_id 'fid' --num_files 1 --model_size small
```

To run REINFORCE with no models:
```
python parser.py --algo reinforce --model_type model_free --max_actions 200 --states_dim 2 --salient_states_dim 2 \
       --extra_dims_stable True --verbose 100 --rs $SLURM_ARRAY_TASK_ID --file_id 'final_smallModel_'$SLURM_ARRAY_TASK_ID
```
And so on. More examples can be found in ```sbatch_reinforce.sh```.

### DDPG (Lillicrap et al. 2017)

To run PAML on HalfCheetah with redundant extra dimensions for example:
```
python parser.py --algo ac --model_type paml --model_size nn --hidden_size 128 --env HalfCheetah-v2 \
               --max_actions 1000 --states_dim 34 --salient_states_dim 17 --num_iters 200 --real_episodes 1 \
                --virtual_episodes 20 --batch_size 500 --num_action_repeats 1 --planning_horizon 10 \
                --initial_model_lr 1e-3 --rhoER 0.5 --noise_type 'redundant' --verbose 5  --rs 2 --file_id 'fid_1'
```
To plot above experiment:
```
python plotting.py --algo ac --env HalfCheetah-v2 --model_type paml --states_dim 34 --salient_states_dim 17 \
               --max_actions 1000 --file_id 'fid' --num_files 1 --model_size nn --hidden_size 128 --planning_horizon 10
```

And so on. More examples for more environments can be found in ```cheetah_sbatch.sh```.

# Author
* **Romina Abachi** - [Github](https://github.com/romina72)

