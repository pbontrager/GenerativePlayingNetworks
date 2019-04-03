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

	#Agent
	num_processes = 8
	agent = PPOAgent(env, num_processes)

	#Training
	t = Trainer(gen, agent, "experiments", 0)
	t.train(4, 32, 1024)
	#t.train(10e6, 8192, 32) #10m training steps, in batches of 8192 steps per 32 levels

if(__name__ == "__main__"):
	main('aliens', 2000)
