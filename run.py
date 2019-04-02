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
	agent = PPOAgent(env, gen, num_processes)

	#Training
	t = Trainer(gen, agent, "experiments", 0)
	t.train(4, 32, 1024)
	#t.train(100, 32, 8000) #100 gen updates in batches of 32, 8 training processes for 8000 steps
	# 32 x 2000 steps x 100 = 6.4m steps
	#Should be process based: 8 x ~1000 steps x 800 = 6.4m steps

if(__name__ == "__main__"):
	main('aliens', 2000)
