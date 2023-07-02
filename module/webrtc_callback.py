import av
from module.rps_game import *

def callback(frame):
	img = frame.to_ndarray(format='bgr24')
	img = computer_vision_game(img)
	return av.VideoFrame.from_ndarray(img, format="bgr24")
