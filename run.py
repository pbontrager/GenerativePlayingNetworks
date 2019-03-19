from models.generator import Generator

import game.game_data as gd
from game.wrappers import GridGame

def main(game_name, game_length):
	#Game description
	mapping = gd.GameDescription[game_name]['mapping']
	ascii = gd.GameDescription[game_name]['ascii']
	state_shape = gd.GameDescription[game_name]['state_shape']
	shapes = gd.GameDescription[game_name]['model_shape']

	#Network
	#gen = None
	latent_shape = (100,)
	gen = Generator(mapping, shapes, latent_shape, state_shape)

	#Environment
	env = GridGame(game_name, game_length, state_shape, ascii, gen)

	#Training
	env.reset()
	score = 0
	for i in range(game_length):
		state, reward, isOver, debug = env.step(env.action_space.sample())
		score += reward
		if(isOver):
			break
	print(score)

if(__name__ == "__main__"):
	main('aliens', 200)
