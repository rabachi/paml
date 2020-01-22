import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb
import argparse
import os

if __name__ == "__main__":
	#Parse input arguments
	parser = argparse.ArgumentParser(description='Arguments available')
	parser.add_argument('--file_location', type=str, default='/scratch/gobi1/abachiro/paml_results',
						help='directory where data is saved')
	parser.add_argument('--file_id', type=str, default='0',
						help='optional additional information to include and for saving multiple runs separately')
	parser.add_argument('--algo', type=str, help='ac or reinforce')
	parser.add_argument('--env', type=str, help='lin_dyn, Pendulum-v0, HalfCheetah-v2, dm-Cartpole-balance-v0')
	parser.add_argument('--model_types', type=str, default='paml',
						help='the model types to plot at once, e.g "paml, mle, model_free" will plot paml performance '
							 'and mle and model-free')
	parser.add_argument('--states_dim', type=int, default=5, help='total states dimensions')
	parser.add_argument('--salient_states_dim', type=int, default=2,
						help='The number of non-random state dimensions, make sure if running ac that the dynamics '
							 'are set correctly')
	parser.add_argument('--model_size', type=str, help='use "small", "constrained", or "nn"')
	parser.add_argument('--hidden_size', type=int, help='hidden_size of model chosen')
	parser.add_argument('--max_actions', type=int, default=20, help='-1 length of trajectory')
	parser.add_argument('--planning_horizon', type=int, default=1, help='planning horizon for actor-critic formulation')
	parser.add_argument('--num_files', type=int, default=10, help='number of runs to plot, these correspond to number '
																  'of random seeds ran during experiments')
	args = parser.parse_args()

	#Check that inputs are correct
	model_types = args.model_types.split(',')
	for mt in model_types:
		if mt not in ['mle', 'paml', 'model_free']:
			raise ValueError('one or more of the model types are not valid entries. '
							 'Only allowed ones are "mle", "paml", "model_free" '
							 'or combinations thereof separated by ","')
	if args.algo not in ['ac', 'reinforce']:
		raise ValueError('Algo must be one of "ac" or "reinforce"')

	font = {'size'   : 16}

	matplotlib.rc('font', **font)

	f_act = plt.figure(figsize=(9,6))
	plt.xlim(0,200)
	plt.ylim(-750, 3250)

	salient_states_dim = args.salient_states_dim
	states_dim = args.states_dim
	traj_length = args.max_actions + 1
	horizon = args.planning_horizon if args.algo == 'ac' else args.max_actions
	model_size = args.model_size
	hidden = args.hidden_size if model_size in ['constrained', 'nn'] else states_dim
	env = args.env if args.algo == 'ac' else 'lin_dyn'
	num_files = args.num_files
	range_num_files = range(num_files)

	skip_5_mb = False
	skip_5_all = False
	if args.algo == 'ac' and env == 'HalfCheetah-v2':
		skip_5_mb = True
	elif args.algo == 'ac' and env == 'Pendulum-v0' or 'dm-Cartpole-balance':
		skip_5_all = True

	colors = {'mle': 'C1', 'paml': 'C2', 'model_free': 'C0'}
	models_all = {}

	algo = args.algo if args.algo == 'reinforce' else 'actorcritic'
	for mt in model_types:
		file_lengths = np.zeros(num_files)
		for index in range_num_files:
			# if mt == 'paml':
			# 	args.file_id = 'fid2'
			# else:
			# 	args.file_id = 'fid'
			if mt == 'model_free':
				horizon_ = 1 if args.algo == 'ac' else horizon
				filename = \
					'{}_state{}_salient{}_rewards_{}_checkpoint_use_model_False_{}_horizon{}_traj{}_{}_{}.npy' \
						.format(mt, states_dim, salient_states_dim, algo, env, horizon_, traj_length,
								args.file_id, index + 1)
			else:
				filename = \
				'{}_state{}_salient{}_rewards_{}_checkpoint_use_model_False_{}_horizon{}_traj{}_{}Model_hidden{}_{}_{}.npy'\
						.format(mt, states_dim, salient_states_dim, algo, env, horizon, traj_length, model_size,
								hidden, args.file_id, index + 1)
			file_lengths[index] = len(np.load(os.path.join(args.file_location, filename)))
			if file_lengths[index] <= 1:
				raise ValueError('File too small, wait for experiment to run longer if still running')
		print(file_lengths)
		num_eps = int(np.min(file_lengths))
		models_all[mt] = np.zeros((num_files, num_eps))
		for index in range_num_files:
			if mt == 'model_free':
				horizon_ = 1 if args.algo == 'ac' else horizon
				filename = \
					'{}_state{}_salient{}_rewards_{}_checkpoint_use_model_False_{}_horizon{}_traj{}_{}_{}.npy' \
						.format(mt, states_dim, salient_states_dim, algo, env, horizon_, traj_length,
								args.file_id, index + 1)
			else:
				filename = \
					'{}_state{}_salient{}_rewards_{}_checkpoint_use_model_False_{}_horizon{}_traj{}_{}Model_hidden{}_{}_{}.npy' \
						.format(mt, states_dim, salient_states_dim, algo, env, horizon, traj_length, model_size,
								hidden, args.file_id, index + 1)
			models_all[mt][index, :] = np.load(os.path.join(args.file_location, filename))[:num_eps]

		if args.algo == 'ac' and (skip_5_all or (skip_5_mb and mt in ['paml', 'mle'])):
			plot_mean = np.mean(models_all[mt], axis=0).squeeze()[::5]
			plot_max = (np.std(models_all[mt],axis=0).squeeze()/ np.sqrt(num_files))[::5]
			plot_min = -plot_max
		elif args.algo == 'reinforce' and (mt in ['model_free']):
			plot_mean = np.mean(models_all[mt], axis=0).squeeze()[::2]
			plot_max = (np.std(models_all[mt],axis=0).squeeze()/ np.sqrt(num_files))[::2]
			plot_min = -plot_max
		else:
			plot_mean = np.mean(models_all[mt], axis=0).squeeze()
			plot_max = (np.std(models_all[mt],axis=0).squeeze()/ np.sqrt(num_files))
			plot_min = -plot_max

		plt.plot(plot_mean, label='{}'.format(mt), color=colors[mt])
		plt.fill_between(np.linspace(0, len(plot_min) - 1, len(plot_min)),
						plot_min + plot_mean, plot_max + plot_mean,
						alpha=0.5, color=colors[mt])

	plt.title('{} irrelevant dims'.format(states_dim - salient_states_dim))
	plt.xlabel('1000 timesteps')
	plt.ylabel('Rewards from true dynamics')
	plt.legend(ncol=3)
	plt.show()

	f_act.savefig(f'images/graph_{algo}_{env}_state{states_dim}.pdf', bbox_inches='tight')
	plt.close(f_act)

