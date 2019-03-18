from setuptools import setup, find_packages

#gvgai_gym using the ascii branch, version not out yet
setup(name='generative_playing_networks',
	version='0.0.1',
	packages= find_packages(),
	install_requires=['gym>=0.10.5', 'numpy>=1.13.3', 'gvgai_gym', 'pytorch', 'pytorch_a2c_ppo_acktr'])
