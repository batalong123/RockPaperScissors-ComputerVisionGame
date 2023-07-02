import cv2, os, random
import numpy as np
import onnx, time
import onnxruntime as ort
import streamlit as st

model_path = 'ai_bot/'
model = os.listdir(model_path)
onnx_path = model_path+model[0]
ort_session = ort.InferenceSession(onnx_path)


@st.cache_resource
class AIBot:
	def __init__(self, uid):
		self.uid = uid

	def choose(self, observation):
		observation = observation.astype(np.float32).reshape(1,-1)
		ort_inputs = {ort_session.get_inputs()[0].name:  observation}
		action = ort_session.run(None, ort_inputs)
		return int(action[0])

@st.cache_resource
class GameBoard:

	def __init__(self):
		pass


	def play_game(self, img=None, prediction_score=None, player_choice=None, current_round=None, hand_in_circle=None, hand_orientation=None, turn_frame=None, fps=None):

		w, h, _ = img.shape

		if prediction_score >= 0.97 and hand_in_circle:

			if self.uid_human == 0:
				self.observation.append(np.array([self.rps_human[player_choice], current_round, sum(self.game_reward)]))
				self.previous_choice_of_human = self.rps_human[player_choice]
			else:
				self.observation.append(np.array([self.previous_choice_of_human, current_round, sum(self.game_reward)]))

			if (hand_orientation > 90 or hand_orientation < -90) and turn_frame%fps == 0:
				obs = np.vstack(self.observation)
				bot_choice = self.agent.choose(obs)
				reward = self.__controller(player_choice=player_choice, bot_choice=bot_choice)
				self.game_reward.append(reward)
				self.observation = []
				return self.rps_bot[bot_choice]

			elif (hand_orientation < -45 and hand_orientation > -70) and turn_frame%fps == 0:
				obs = np.vstack(self.observation)
				bot_choice = self.agent.choose(obs)
				reward = self.__controller(player_choice=player_choice, bot_choice=bot_choice)
				self.game_reward.append(reward)
				self.observation = []
				return self.rps_bot[bot_choice]

			elif (hand_orientation < -45 and hand_orientation > -90) and turn_frame%fps == 0:
				obs = np.vstack(self.observation)
				bot_choice = self.agent.choose(obs)
				reward = self.__controller(player_choice=player_choice, bot_choice=bot_choice)
				self.game_reward.append(reward)
				self.observation = []
				return self.rps_bot[bot_choice]
			else:
				self.observation = []
				return 

		else:
			return 

	def reset_game(self):

		self.previous_choice_of_human = random.choice(range(3))
		self.bot_choice =self.previous_choice_of_human
		self.observation = []


		self.human_score = 0 
		self.bot_score = 0
		self.nbr_of_draw = 0
		self.winner = " "
		self.rps_human = {'rock':0, 'paper':1, 'scissors':2}
		self.rps_bot = {0:'rock', 1:'paper',  2:'scissors'}

		self.current_round = random.choice([5, 11, 21])
		self.game_reward = []
		self.reward = None
		self.uid_human = random.choice([0,1])
		self.agent = AIBot((self.uid_human+1)%2)

		return self.current_round
		

	def __controller(self, player_choice=None, bot_choice=None):

		if self.rps_human[player_choice] == bot_choice:
			self.winner = 'tie'
			self.nbr_of_draw += 1
			self.reward = 0

		elif self.rps_human[player_choice] - bot_choice == 1 or self.rps_human[player_choice] - bot_choice == -2:
			self.human_score += 1
			self.winner = 'human'
			self.reward = -1
		else:
			self.bot_score += 1
			self.winner = 'bot'
			self.reward = 1
		return self.reward

	def game_message(self):
		return self.human_score, self.bot_score, self.nbr_of_draw

