from module.visions import (board, gestures_recognizer, annotate, start_game,end_game,textMenu)
from module.game_utils import GameBoard
import cv2, pygame
import av
import streamlit as st
from streamlit_webrtc import VideoProcessorBase
from av.video.frame import VideoFrame


class ComputerVisionGame(VideoProcessorBase):
	
	def __init__(self):
		super(ComputerVisionGame, self).__init__()

		sound_path = 'sound/'
		pygame.init()
		pygame.mixer.init()
		pygame.mixer.music.load(os.path.join(sund_path, 'cam.mp3'))

		fps = 30.0
		self.FPS = fps//3

		images_path = 'images/'
		self.img_filenames = {os.path.splitext(u)[0]: os.path.join(images_path, u) for u in os.listdir(images_path)}

		self.game_running = False
		self.game_state = "start"
		self.game = GameBoard()
		self.turn_time = 0; self.response_time = 15; self.current_rounds = None;

		self.human_hand = ''; self.bot_hand = ''

	def recv(self, frame):

		img = frame.to_ndarray(format='bgr24')
		img = self.__display_game(img)
		
		return VideoFrame.from_ndarray(img, format='bgr24')

	def __display_game(self, img):

		img_height, img_width, _ = img.shape

		#img = IEHC(img)

		img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)
		#print('Is rps game vision running?', self.game_running)

		if self.game_running:

			img = board(img, img_width, img_height)
			img, sign, score, hand_in_circle, angle = gestures_recognizer(img)

			if (sign, score) != (None, None) and str(sign) not in ['', 'none']:
				cv2.putText(img, f"Prediction: {sign} ({100*score:.2f}%)", (10, img_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 120, 0), 2)
				bot_choice = self.game.play_game(img= img, prediction_score=score, player_choice=sign, current_round=self.current_rounds, 
					hand_in_circle=hand_in_circle, hand_orientation=angle, turn_frame=self.step, fps=self.FPS)

				#update 
				if bot_choice:
					pygame.mixer.music.play()
					img = annotate(img, [self.img_filenames[sign], self.img_filenames[bot_choice]])
					self.human_hand = sign
					self.bot_hand = bot_choice
					self.current_rounds -= 1


				self.turn_time = 0
				self.response_time = 15
				#current_rounds -= int(step%FPS == 0)
				self.game_running = (self.current_rounds > 0)
				if not self.game_running:
					self.game_state = "game_over"
					self.final_human_score , self.final_bot_score = self.human_score, self.bot_score

				self.human_score, self.bot_score, self.tie = self.game.game_message()

			else:
				self.turn_time += 1
				self.response_time -= int(self.turn_time%6 == 0)

			
			cv2.putText(img, f"Time_down(in sec): {self.response_time}", (10, img.shape[1] - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)
			cv2.putText(img, f"Round_down: {self.current_rounds}", (img.shape[0]-110, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (55, 200, 0), 2)
			if self.response_time==0:
		
				self.bot_score += 1
				self.current_rounds -= 1
				self.response_time = 15
				self.game_running = (self.current_rounds > 0)
				if not self.game_running:
					self.game_state = "game_over"
					self.final_human_score , self.final_bot_score = self.human_score, self.bot_score
				self.answer = 'Yes'; self.count_forfeit += 1

			cv2.putText(img, f"Human score: {self.human_score} pts", (img.shape[0]-110, 45), cv2.FONT_HERSHEY_PLAIN, 1.05, (55, 200, 0), 2)
			cv2.putText(img, f"AI Bot score: {self.bot_score} pts", (img.shape[0]-110, 65), cv2.FONT_HERSHEY_PLAIN, 1.05, (55, 200, 0), 2)
			cv2.putText(img, f"Nbr of ties: {self.tie}", (img.shape[0]-110, 85), cv2.FONT_HERSHEY_PLAIN, 1.05, (55, 200, 0), 2)
			cv2.putText(img, f"Forfeit: {self.answer}({self.count_forfeit}).", (img.shape[0]-110, 105), cv2.FONT_HERSHEY_PLAIN, 1.05, (255, 0, 0), 2)
			cv2.putText(img, f"Human choice: {self.human_hand}.", (img.shape[0]-110, 125), cv2.FONT_HERSHEY_PLAIN, 1.05, (255, 55, 255), 2)
			cv2.putText(img, f"AI Bot choice: {self.bot_hand}.", (img.shape[0]-110, 145), cv2.FONT_HERSHEY_PLAIN, 1.05, (255, 55, 255), 2)

			
		else:
			if self.game_state == "start":
				img = start_game(img) 

			if self.game_state == "game_over":
				img = end_game(img)

				if self.final_human_score > self.final_bot_score:
					img = textMenu(img, text="YOU WON", color=(0, 255, 0),
					font_scale=0.775, thickness=2, height=150)
					img = textMenu(img, text=f"Human {self.final_human_score} vs {self.final_bot_score} AI Bot.", color=(0, 255, 0),
					font_scale=0.775, thickness=2, height=190)
				elif self.final_human_score < self.final_bot_score:
					img = textMenu(img, text="YOU LOST", color=(255, 0, 0),
					font_scale=0.775, thickness=2, height=150)
					img = textMenu(img, text=f"Human {self.final_human_score} vs {self.final_bot_score} AI Bot.", color=(255, 0, 0),
					font_scale=0.775, thickness=2, height=190)
				else:
					img = textMenu(img, text="YOU TIED", color=(205, 205, 0),
					font_scale=0.775, thickness=2, height=150)
					img = textMenu(img, text=f"Human {self.final_human_score} vs {self.final_bot_score} AI Bot.", color=(205, 205, 0),
					font_scale=0.775, thickness=2, height=190)



			self.current_rounds =  self.game.reset_game()
			self.step = 1
			self.human_score, self.bot_score, self.tie = 0, 0, 0
			self.answer = 'No'; self.count_forfeit = 0

		self.step += 1
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		return img
