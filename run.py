from models.generator import Generator
from agents.ppoagent import PPOAgent
from trainer import Trainer
from game.env import Env


def main(game_name, game_length):
	#Game description
	env = Env(game_name, game_length)

	#Network
	latent_shape = (100,)
	gen = Generator(latent_shape, env)

        #Here: verify agents reward going the right way
	#Run without generator to see training
        #Focus on doing better than random initialization (maybe don't update generator)

	#Agent
	num_processes = 24
	experiment = "experiment_rl"
	agent = PPOAgent(env, num_processes, experiment) #, lr=.001)

	#Training
	t = Trainer(gen, agent, experiment, 0) #1 for pretrained
	t.train(1000, 512, 1e8) #1000, 32, 8192

if(__name__ == "__main__"):
	main('zelda', 1000)
