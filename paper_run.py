from models.generator import Generator
from agents.agent import Agent
from trainer import Trainer
from game.env import Env
import torch


def main(game_name, game_length):
	#Game description
	reward_mode = 'time'
	reward_scale = 1.0
	elite_prob = .5
	env = Env(game_name, game_length, {'reward_mode': reward_mode, 'reward_scale': reward_scale, 'elite_prob': elite_prob})

	#Network
	latent_shape = (512,)
	dropout = .2
	lr = .0001
	gen = Generator(latent_shape, env, 'pixel', dropout, lr)

	#Agent
	num_processes = 16
	experiment = "Experiment_Paper"
	lr = .00025
	model = 'resnet'
	dropout = 0
	reconstruct = gen
	r_weight = .05
	Agent.num_steps = 5
	Agent.entropy_coef = .01
	Agent.value_loss_coef = .1
	agent = Agent(env, num_processes, experiment, 0, lr, model, dropout, reconstruct, r_weight)

	#Training
	gen_updates = 100
	gen_batch = 128
	gen_batches = 10
	diversity_batches = 90
	rl_batch = 1e6
	pretrain = 2e7
	elite_persist = True
	elite_mode = 'max'
	load_version = 0
	notes = 'Configured to match paper results'
	agent.writer.add_hparams({'Experiment': experiment, 'RL_LR':lr, 'Minibatch':gen_batch, 'RL_Steps': rl_batch, 'Notes':notes}, {})
	t = Trainer(gen, agent, experiment, load_version, elite_mode, elite_persist)
	t.train(gen_updates, gen_batch, gen_batches, diversity_batches, rl_batch, pretrain)

if(__name__ == "__main__"):
	main('zelda', 1000)
