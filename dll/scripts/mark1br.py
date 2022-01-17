'''
UNDER MIT OPENSOURCE LICENSE. IG?
'''
from godot import exposed, export, Area2D
from godot import *
import numpy as np
import torch
import random
import torch.nn as nn
from collections import deque
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle


import os
import sys
os.environ["TCL_LIBRARY"] = os.path.join(sys.base_exec_prefix, "lib", "tcl8.6")


# plt.ion()
plt.style.use('ggplot')





# The convolutional block
def nature_cnn(observation_space, depth =(1, 64, 64), final_layer = 512):
	n_input_channel = observation_space.shape[0]

	cnn = nn.Sequential(
		nn.Conv1d(n_input_channel, depth[0], kernel_size = 1),
		nn.ReLU(),
		nn.Conv1d(depth[0], depth[1], kernel_size = 1),
		nn.ReLU(),
		nn.Conv1d(depth[1], depth[2], kernel_size = 1),
		nn.ReLU(),
		nn.Flatten(),
		)

	with torch.no_grad():
		t = torch.reshape(torch.as_tensor(observation_space.sample()[None]).float(), (-1, n_input_channel, 1))
		n_flatten = cnn(t).shape[1]

	out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

	return out 

# The action class


class ActionSpace:
	def __init__(self):
		self.n = 5 
		# Right(1,0), left(-1,0), jump(0,-1), do-nothing(0,0) and Attack(1,1)
		self.actions = [Vector2(1,0),Vector2(-1,0),Vector2(0, -1), Vector2(0,0), Vector2(1,1)]
	def sample(self):
		return random.randint(0,self.n-1) # *self.SPEED




# The Observation Class
class ObservationSpace:
	def __init__(self, target_player, itself):
		self.target = target_player
		self.itself = itself
		self.shape = (3,)

	# RUNDOWN-1 The sample() func should return the distance betweent the target player and the agent.
	# RUNDOWN-2: More observations should be returned such as target player hitbox area distance etc?
	# Currently returns the position of the player as observation
	
	def get_distance_to(self, target_x, target_y):
		# converting the player/target co-ordinates to numpy array.
		cord_player = np.array([target_x, target_y])
		# converting the self/agent co-ordinates to numpy array.
		cord_itself = np.array([self.itself.position.x, self.itself.position.y])
		# calculating the euclidean distance between the target and the agent
		normal_dist = np.linalg.norm(cord_player- cord_itself)

		# returning the calculated distance
		return normal_dist

	def sample(self):
		# returning the distance between the agent and the target using the get_distance_to function and turning it into an array/list
		# return [self.get_distance_to(self.target.position.x, self.target.position.y)]
		# returning the actual co-ordinates of the target
		return np.array([self.target.position.x, self.target.position.y, self.get_distance_to(self.target.position.x, self.target.position.y)])


class Env:
	# initializes the values that are necessary for the enviroment to have access to /
	# interms of executing auxilary functions such as step() and reset()
	def __init__(self,pivot, sprite, animationPlayer, player, itself, brain, detection_radius, training = True):


		# boolean that sets the environment to training mode if true
		self.training = training

		# The target player/ human player object
		self.player = player

		# The agent itself(Main Kinematicbody)
		self.itself = itself
		# The agent's brain
		self.brain = brain

		# Agent animationplayer
		self.animationPlayer = animationPlayer

		# Agent sprite
		self.sprite = sprite

		# Agent weapon pivot
		self.pivot = pivot

		# Observation Class instance that takes in the target player/human player object /
		# returns the distance between the player and the agent itself(still to do) 
		self.observation_space = ObservationSpace(self.player, self.itself)
		
		# Action Space class instance that returns a random movement if sample() func is used
		# contains possible movements info.
		self.action_space = ActionSpace()
		

		self.SPEED = 800
		self.JUMP_HEIGHT = 950
		self.GRAVITY = 40
		self.detection_radius = detection_radius
		self.done = False

		self.motion = Vector2.ZERO





	# Resets the environment after each session/episode ends or if the agent dies
	# It resets the position of the agent and the player in the environment
	def reset(self):
		if self.training:
			# For now(Rundown 1) we just reset the agent's position as the player is static
			self.itself.position = self.brain.initial_position
			self.brain.start_episode()
		return self.observation_space.sample()
	


	# Auxilary fucntion that manages the direction the agent looks and also flips the hurtbox corresponding to the movement.
	def manage_agent_look(self, flip):
		# flipping the sprite of the agent and rotating the hurtbox if the flip is on
		if flip:
			self.sprite.flip_h = True
			self.pivot.rotation_degrees = 180
		# Flipping the sprite of the agent and the hurtbox to normal directions if the flip is off
		else:
			self.sprite.flip_h = False
			self.pivot.rotation_degrees = 0


	# Step function returns msotion of the agent, new_observation, reward and session_complete_state(done)
	def step(self, action):
		# Applying gravity.
		self.motion.y+=self.GRAVITY

		# managing the horizontal movements.
		self.motion.x = action.x*self.SPEED


		# check if the player is on floor(block)
		if self.itself.is_on_floor():

			# Creating attack movement

			# TODO: The attack isn't completing(Animation, which also controls the hurtbox enable)

			if action == Vector2(1,1):
				self.animationPlayer.play("Attack")
				self.motion = Vector2(0,0)


			# make sure that other movements aren't done while attacking\n
			# Movement can only be done if the agent isn't attacking.
			else:
				# Jumping!
				if action.y == -1:
					self.motion.y = -self.JUMP_HEIGHT

				# Playing the ground movement animation
				if self.motion.x>0:
					self.animationPlayer.play("Run")
					self.manage_agent_look(False)
				else:
					self.animationPlayer.play("Run")
					self.manage_agent_look(True)




			

		# if the player isn't on floor(/is in air) block
		else:
			# play jumping animation
			if self.motion.y<0:
				self.animationPlayer.play("Jump")
			else:
				self.animationPlayer.play("Fall")


		# Manage agent look block and correcting the direction of the hurtbox corresponding to the movement\n
		# using the method manage_agent_look
		if self.motion.x>0:
			self.manage_agent_look(False)
		else:
			self.manage_agent_look(True)


		# Getting the observation
		observation = self.observation_space.sample()

		# Calculating the negative reward depending on how close the target is.
		# reward = float("{:.2f}".format( - (observation[0]/self.detection_radius)))

		# calculating the reward when actual co-ordinate of the target is used as feature
		# -> calculating the euclidean distance to the player(target) using the auxilary function from the observation class
		obs_dist = self.observation_space.get_distance_to(observation[0], observation[1])
		reward = float("{:.2f}".format( - (observation[2]/self.detection_radius)))


		# returning the requirements to the main loop to initialize and train upon.
		return observation, reward, self.done, self.motion
		




class Brain(nn.Module):
	def __init__(self, env, use_cnn = False):
		# calling the init method of the parent class, i.e nn.Module
		super().__init__()
		if not use_cnn:

			# Calculating the shape of the input nodes/neurons
			in_features = int(np.prod(env.observation_space.shape))

			# Creating/defining the feed-forward neural network(vanilla version)\n
			# with 2 Linear layers and a Tanh activation function.
			self.net = nn.Sequential(
				nn.Linear(in_features, 64),
				nn.Tanh(),
				nn.Linear(64, 128),
				nn.Tanh(),
				nn.Linear(128, 64),
				nn.Tanh(),
				nn.Linear(64, env.action_space.n)
				)
		else:
			self.num_actions = env.action_space.n
			conv_net = nature_cnn(env.observation_space)
			self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))



	def forward(self, x):
		# passing the observation through the neural-network(forward-pass)
		return self.net(x)



	def act(self, obs):
		# Converting the observations to torch.tensor
		obs_t = torch.as_tensor(obs, dtype = torch.float32)
		# calculating the q_values.
		q_values = self(obs_t.unsqueeze(0))
		# getting the index of max q value
		max_q_index = torch.argmax(q_values, dim = 1)[0]
		# getting the action depending on the index of the max q index.
		action = max_q_index.detach().item()
		# returning the action that the neural net thinks to be the best.
		return action


@exposed
class mark1br(Area2D):
	# Auxiliary functions
	def initialize_env(self, body=None,initialize = True):
		if initialize:
			self.env = Env(self.pivot ,self.sprite, self.animationPlayer, player = body, itself = self.itself, brain=self, detection_radius = self.detection_radius)
			# self.start_episode()
		else:
			self.env = None

	def end_episode(self):
		if self.env is not None:
			self.env.done = True


	def start_episode(self):
		if self.env is not None:
			self.env.done = False
			self.done = self.env.done
			self.start_timer()

	def start_timer(self):
		# Starts the episode timer for declared amount of time
		self.timer.wait_time = self.episode_timeout
		self.timer.start(self.episode_timeout)


	# Signals
	def _on_player_entered(self, body):
		self.initialize_env(body = body)

	def _on_player_exited(self, body):
		self.initialize_env(initialize = False)

	def _on_hitbox_area_entered(self, area):
		self.episode_reward+=5

	def _on_reset_timer_end(self):
		self.end_episode()

	def _on_health_zero(self):
		self.end_episode()

	def _on_hurtbox_area_entered(self, area):
		self.stats.health -= area.damage
		print(self.stats.health)

	# Godot-reserved functions,
	def _ready(self):
		# agent parameters
		self.movement_vectors = [Vector2(1,0),Vector2(-1,0),Vector2(0, -1), Vector2(0,0), Vector2(1,1)]
		self.SPEED = 800
		self.JUMP_HEIGHT = 950
		self.GRAVITY = 40
		self.stats = self.get_parent().get_node('EnemyStats')
		self.timer = self.get_parent().get_node('Timer')
		self.env = None
		self.itself = self.get_parent()
		self.initial_position = self.itself.position
		self.detection_radius = self.get_node("CollisionShape2D").get_shape().radius
		self.episode_timeout = 10
		self.motion = Vector2.ZERO
		self.animationPlayer = self.get_parent().get_node("AnimationPlayer")
		self.sprite = self.get_parent().get_node('Sprite')
		self.pivot = self.get_parent().get_node("Position2D")



		# Agent model-hyperparameters.
		self.GAMMA=0.99
		self.BATCH_SIZE=16
		self.BUFFER_SIZE=5000 # 50000
		self.MIN_REPLAY_SIZE=500 # 1000
		self.EPSILON_START=1.0
		self.EPSILON_END=0.02
		self.EPSILON_DECAY=5000 # 10000
		self.TARGET_UPDATE_FREQ = 1000
		self.lr = 5e-4

		self.SAVE_INTERVAL = 10000
		self.LOG_INTERVAL = 1000

		# Training essentials
		self.replay_buffer = deque(maxlen = self.BUFFER_SIZE)
		self.rew_buffer = deque([0.0], maxlen = 100)
		self.episode_reward = 0.0
		self.done = False

		self.step = 0

		self.initialize = True
		self.first_time = True


		# Graph axis:
		# x - step
		# y - reward, cumulative average.
		self.generate_graph = True

		self.graph_generation_interval = 1000
		# self.step_x = []
		# self.reward = []
		# self.avg_reward = []

		# self.loss = []
		# self.cumulative_loss = []
		self.generation = 0
		

		# Model saving parameters
		self.episode_datafile_path = "/home/ishraque/Desktop/startUp/dll/model_data/model_episode_data/data_deep_4_layers.pickle"
		self.graph_path = "/home/ishraque/Desktop/startUp/dll/model_data/graph/deep4progression.png"
		self.model_path = "/home/ishraque/Desktop/startUp/dll/model_data/model_pickle/" 
		self.loss_graph = "/home/ishraque/Desktop/startUp/dll/model_data/graph/deep4model_loss.png"
	def _physics_process(self, delta):
		# We always need to check if self.env != None before we start an episode or call any functions related.
		if self.env is not None:
			if self.initialize:
				# t = torch.reshape(torch.as_tensor(self.env.observation_space.sample()).float(), (-1,3,1))
				# print(t.shape)

				print("Starting buffer initialization...")
				# neural networks
				self.online_net = Brain(self.env, use_cnn = True)
				self.target_net = Brain(self.env, use_cnn = True)
				
				# load the neural network from a pickled file
				# with open(self.model_path+"Model_Gen-372.pickle", 'rb') as f:
				# 	self.online_net = pickle.load(f)

				# making the target net parameters same/equal to the online net
				self.target_net.load_state_dict(self.online_net.state_dict())
				self.optimizer = optim.Adam(self.online_net.parameters(), self.lr)

				# Initialize replay buffer
				self.obs = self.env.reset()
				for _ in range(self.MIN_REPLAY_SIZE):
					action_i = self.env.action_space.sample()
					action = self.movement_vectors[action_i] 

					new_obs, rew, self.done, _ = self.env.step(action)
					transition = (self.obs, action_i, rew, self.done, new_obs)
					self.replay_buffer.append(transition)
					self.obs = new_obs

					if self.done:
						self.obs = self.env.reset()

				print("Initialization Completed...\nStarting Training...")
				self.obs = self.env.reset()
				self.initialize = False


			#Main Training block
			self.step+=1
			action = Vector2.ZERO
			if not self.done:
				epsilon = np.interp(self.step, [0, self.EPSILON_DECAY], [self.EPSILON_START, self.EPSILON_END])

				rnd_sample = random.random()
				if rnd_sample<=epsilon:
					action_i = self.env.action_space.sample()
					action = self.movement_vectors[action_i]
				else:
					action_i = self.online_net.act(self.obs)
					action = self.movement_vectors[action_i]

				new_obs, rew, self.done, self.motion = self.env.step(action)
				transition = (self.obs, action_i, rew, self.done, new_obs)
				self.replay_buffer.append(transition)
				self.obs = new_obs
				self.episode_reward+= rew

				self.motion = self.itself.move_and_slide(self.motion, Vector2.UP)


				# When done with the episode!
				if self.done:
					self.obs = self.env.reset()
					# print(f"Episode ended, episode reward: {self.episode_reward}")
					self.rew_buffer.append(self.episode_reward)
					self.episode_reward = 0
					self.generation +=1

					# Randomixing the initial_position
					self.initial_position = Vector2(random.randint(500,1400),random.randint(-50, 0))



				transitions = random.sample(self.replay_buffer, self.BATCH_SIZE)

				obses = np.asarray([t[0] for t in transitions])
				actions = np.asarray([t[1] for t in transitions])
				rews = np.asarray([t[2] for t in transitions])
				dones = np.asarray([t[3] for t in transitions])
				new_obs = np.asarray([t[4] for t in transitions])



				# converting to tensors
				obses_t = torch.as_tensor(obses, dtype = torch.float32)
				actions_t = torch.as_tensor(actions, dtype = torch.int64).unsqueeze(-1)
				rews_t = torch.as_tensor(rews, dtype = torch.float32).unsqueeze(-1)
				dones_t = torch.as_tensor(dones, dtype = torch.float32).unsqueeze(-1)
				new_obs_t = torch.as_tensor(new_obs, dtype = torch.float32)

				target_q_values = self.target_net(new_obs_t)
				max_target_q_values =  target_q_values.max(dim = 1, keepdim = True)[0]


				targets = rews_t+ self.GAMMA * (1-dones_t) * max_target_q_values

				q_values = self.online_net(obses_t)
				action_q_values = torch.gather(input = q_values, dim = 1, index = actions_t)

				loss = nn.functional.smooth_l1_loss(action_q_values, targets)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

				# Update the target network/offline network with the online network's node/neurone parameters(weights and biases)
				if not self.step%self.TARGET_UPDATE_FREQ:
					self.target_net.load_state_dict(self.online_net.state_dict())



				# Log training progression and backing up the progression graph data 
				if not self.step%self.LOG_INTERVAL and self.step:
					print(f'Generation:{self.generation}\nstep: {self.step}\nAvg Rew: {np.mean(self.rew_buffer)}\n episode reward: {self.rew_buffer[-1]}')
					
					

				if self.generate_graph:
					# Save the training progression data/values(more frequent than the logging rate)
					if not self.step%self.graph_generation_interval and self.step:
						# self.loss.append(loss.item())
						# self.cumulative_loss.append(np.mean(self.loss))

						# self.step_x.append(self.step)
						# self.reward.append(self.rew_buffer[-1])          
						# self.avg_reward.append(np.mean(self.rew_buffer))


						# we first check if a datafile already exists or not
						# if it exists we read in the existing data first.
						if os.path.isfile(self.episode_datafile_path) and not self.first_time:
							print("reading the file")
							with open(self.episode_datafile_path, 'rb') as datafile:
								# loading in the existing data
								step_x,rew_y,avg_y = pickle.load(datafile)

								# appending to it
								step_x.append(self.step)
								rew_y.append(self.rew_buffer[-1])
								avg_y.append(np.mean(self.rew_buffer))
						else:
							with open(self.episode_datafile_path, 'wb') as datafile:
								step_x =  [self.step]
								rew_y = [self.rew_buffer[-1]]
								avg_y = [np.mean(self.rew_buffer)]
								pickle.dump([step_x,rew_y, avg_y], datafile)
								self.first_time = False



						# Assuming that the above code has been executed, there must be file now\n
						# and we can write the newly appended data to the file
						with open(self.episode_datafile_path, 'wb') as datafile:
							# write the renewed data to the existing file.
							pickle.dump([step_x, rew_y, avg_y], datafile)



						# print(step_x, rew_y, avg_y)


						plt.xlabel("Steps")
						plt.ylabel("(Average/)Reward")
						plt.plot(step_x, rew_y, label = "Reward")
						plt.plot(step_x, avg_y, label = "Average reward")
						plt.legend(loc="best", shadow = True)
						plt.savefig(self.graph_path)
						plt.close()


						# The loss graph isn't necessarily useful, so we comment it out

						# plt.xlabel("Steps")
						# plt.ylabel("Model Loss")
						# plt.plot(self.step_x, self.loss, label = "Model Loss")
						# plt.plot(self.step_x, self.cumulative_loss, label = "Cumulative Loss")
						# plt.legend(loc = "best", shadow = True)
						# plt.savefig(self.loss_graph)
						# plt.close()

						print(f"Graph plotted at step {self.step}")
						
						# removing the values of the graph data from the memory
						step_x = None
						rew_y = None
						avg_y = None


					


				# TODO: Save the model as pickle file at certain intervals(greater than both log and progression graph render)
				if not self.step%self.SAVE_INTERVAL and self.step:
					print(f"Saving Model, Generation{self.generation}\n")
					with open(self.model_path+f"Model_Gen-{self.generation}-Steps-{self.step}.pickle", 'wb') as f:
						pickle.dump(self.online_net,f)




				# RESUME: WILL THINK ABOUT: Add negative reward for moving too much/ for each move we add negative reward\n
				# this hopefully smoothens and optimizes the movements further.
				
		else:
			self.animationPlayer.play("Idle")