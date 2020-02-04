from models.generator import Generator
from agents.ppoagent import PPOAgent
from trainer import Trainer
from game.env import Env
import torch


def main(game_name, game_length):
	#Game description
	env = Env(game_name, game_length)

	#Network
	latent_shape = (512,)
	gen = Generator(latent_shape, env)

	#Agent
	num_processes = 16
	experiment = "Experiments" #Pretraining set to 0, 20k steps, remove debug, remove wrapper and trainer debug
	agent = PPOAgent(env, num_processes, experiment, lr=.00025, reconstruct=gen) #.00025

	agent.writer.add_hparams({'Experiment': experiment, 'lr':.00025, 'Minibatch':32, 'RL_Steps': 1e5, 'Notes':' '}, {})

	#Training
	t = Trainer(gen, agent, experiment, 0) #save agent_1.tar as pretrained_agent.tar
	t.train(1000, 32, 1e5) #1000, 32, 8192

if(__name__ == "__main__"):
	main('zelda', 1000)
