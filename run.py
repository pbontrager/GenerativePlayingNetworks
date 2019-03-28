from models.generator import Generator
from agents.ppoagent import PPOAgent
from trainer import Trainer
from game.env import Env


def main(game_name, game_length):
	#Game description
	env = Env(game_name, game_length)


	# mapping = gd.GameDescription[game_name]['mapping']
	# ascii = gd.GameDescription[game_name]['ascii']
	# state_shape = gd.GameDescription[game_name]['state_shape']
	# shapes = gd.GameDescription[game_name]['model_shape']

	#Network
	latent_shape = (100,)
	gen = Generator(latent_shape, env)

	#Environment
	# num_processes = 8
	# env_desc = (game_name, game_length, state_shape, ascii, gen)
	# env = make_vec_envs(env_desc, PPOAgent.seed, num_processes, PPOAgent.gamma, PPOAgent.log_dir, PPOAgent.device, False)

	#Agent
	num_processes = 1 #8
	agent = PPOAgent(env, gen, num_processes)

	#Training
	t = Trainer(gen, agent, "experiments", 0)
	t.train(1, 32, 200)
	#t.train(100, 32, 8000) #100 gen updates in batches of 32, 8 training processes for 8000 steps
	# 32 x 2000 steps x 100 = 6.4m steps
	#Should be process based: 8 x ~1000 steps x 800 = 6.4m steps

if(__name__ == "__main__"):
	main('aliens', 200)
