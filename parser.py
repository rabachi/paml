import pdb
import actor_critic
import reinforce

import argparse


if __name__ == "__main__":
	#Parse input arguments and call the reinforce or actor_critic files based on the planner the user intends to use
	parser = argparse.ArgumentParser(description='Arguments available')
	
	parser.add_argument('--algo', type=str, help='ac or reinforce')
	parser.add_argument('--env', type=str, help='lin_dyn or Pendulum-v0')
	parser.add_argument('--model_type', type=str, default='paml', help='paml, mle, model_free, random')
	parser.add_argument('--ensemble', type=int, default=1, help='number of models in ensemble')
	parser.add_argument('--batch_size', type=int, default=64, help='batch_size for actor critic')
	parser.add_argument('--states_dim', type=int, default=5, help='total states dimensions')
	parser.add_argument('--salient_states_dim', type=int, default=2,
		help='The number of non-random state dimensions, make sure if running ac that the dynamics are set correctly')
	parser.add_argument('--extra_dims_stable', help='set to true to have irrelevant dimensions that have stable dynamics')
	parser.add_argument('--model_size', type=str, help='use "small", "constrained", or "nn"')
	parser.add_argument('--hidden_size', type=int, default=128 , help='hidden_size of model chosen')
	parser.add_argument('--initial_model_lr', type=float, default=1e-4)
	parser.add_argument('--num_iters', type=int, default=100, help='Number of gradient updates on model')
	parser.add_argument('--verbose', type=int, default=10)
	parser.add_argument('--max_torque', default=2.0, help='max torque for linear dynamics in only reinforce')
	parser.add_argument('--real_episodes', type=int, default=10, help='number of real episodes to collect between planning updates')
	parser.add_argument('--num_eps_per_start', type=int, default=1)
	parser.add_argument('--virtual_episodes', type=int, default=1000, help='number of virtual episodes to gather from model for planning')
	parser.add_argument('--max_actions', type=int, default=20, help='-1 length of trajectory')
	parser.add_argument('--num_action_repeats', type=int, default=1, help='number of times to repeat an action')
	parser.add_argument('--planning_horizon', type=int, default=1, help='planning horizon for actor-critic fomulation')
	parser.add_argument('--rhoER', type=float, default=0.5, help='fraction of data to use from experience replay in planning')
	parser.add_argument('--discount', default=0.99)
	#continuous action-space, default=True
	parser.add_argument('--file_location', type=str, default='/scratch/gobi1/abachiro/paml_results',
						help='directory for saving training data and checkpoints')
	parser.add_argument('--file_id', type=str, default='0',
						help='optional additional information to include and for saving multiple runs separately')
	parser.add_argument('--rs',type=int, default='0', help='set random seed to ensure uniform intial points for better comparison')
	parser.add_argument('--save_checkpoints_training', help='set to false to save space when running lots of experiments')
	parser.add_argument('--noise_type', default='random', help='if "redundant" use redundant type of noise if states_dim correct, otherwise use random type of noise')
	args = parser.parse_args()


	if args.algo == 'ac':
		actor_critic.main(
							args.env, 
							args.real_episodes,
							args.virtual_episodes,
							args.num_eps_per_start,
							args.num_iters,
							args.max_actions,
							args.discount,
							args.states_dim,
							args.salient_states_dim,
							args.initial_model_lr,
							# initial_policy_lr,
							args.model_type,
							args.file_location,
							args.file_id,
							args.save_checkpoints_training,
							args.batch_size,
							args.verbose,
							args.model_size,
							args.num_action_repeats,
							args.rs,
							args.planning_horizon,
							args.hidden_size,
							args.rhoER,
							args.ensemble,
							args.noise_type
						)
	elif args.algo == 'reinforce':
		reinforce.main(
						args.max_torque,
						args.real_episodes,
						args.virtual_episodes,
						args.num_eps_per_start,
						args.num_iters,
						args.discount,
						args.max_actions,
						args.states_dim,
						args.salient_states_dim,
						args.initial_model_lr,
						args.model_type,
						args.file_location,
						args.file_id,
						args.save_checkpoints_training,
						args.verbose,
						args.extra_dims_stable,
						args.model_size,
						args.rs
					)







