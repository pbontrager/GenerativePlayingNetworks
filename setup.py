from setuptools import setup, find_packages

#gvgai_gym using the ascii branch, version not out yet
setup(name='generative_playing_networks',
	version='0.0.1',
	packages= find_packages(),
	install_requires=['gym>=0.10.5', 'numpy>=1.13.3', 'tensorboard==1.14', 'gvgai_gym','tensorboardX', 'pandas'])

#Install Manually
#Pytorch

#Install from github
#baselines
#pytorch_a2c_ppo_acktr
#gvgai gym -git checkout ascii
