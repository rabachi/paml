# paml
Experiments for policy-aware model learning  
  
Main files to look at are:  
1. single_episode_mle_multi_step.py : MLE training  
2. single_episode_paml_skip.py: PAML training  
  
**R_range** sets horizon of unrolling  
**max_actions** is the number of actions in trajectory (length of trajectory is max_actions + 1)  
**num_episodes**: number of episodes to sample  
**batch_size**: number of episodes to use per batch (sampled without replacement)  
* in PAML: **num_iters**: number of times to sample from data   
           **opt_step_def**: number of optimizations to do per batch of data  
* in MLE: **num_iters**: number of times to sample from data and optimize (only one optimization done per batch)  
  
Ignore the device setting for now, only tested on CPU. GPU not necessary for these experiments.  
  
In models.py, only ReplayMemory, Policy and DirectEnvModel are used.  
In utils.py, only discount_rewards, lin_dyn, shift, and get_selected_log_probabilities are used.  
  
Not fully tested for discrete action-spaces, but extension easy. 
