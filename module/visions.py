import cv2
import os, sys, math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import streamlit as st


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

class Recognizer:
	def __init__(self):
		self.recognition = GestureRecognizerResult(gestures=[], handedness=[], hand_landmarks=[], hand_world_landmarks=[])
	def get_result(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
		#print(result)
		self.recognition = result
		

func = Recognizer()


model_path = 'models/gesture_recognizer_augmented_2.task'

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM, result_callback=func.get_result,
    min_tracking_confidence=0.55, min_hand_presence_confidence=0.55, 
    min_hand_detection_confidence=0.55)

@st.cache_data
def IEHC(image):

	denoised_image = cv2.medianBlur(src=image, ksize=3) #cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 10)
	image = cv2.normalize(denoised_image, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)

	#kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
	#sharpened_image = cv2.filter2D(contrast_stretched_image, -5, kernel=kernel)

	brightness_image = cv2.convertScaleAbs(image, alpha=1, beta=5)

	gamma = 1.0
	lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	gamma_corrected_image = cv2.LUT(brightness_image, lookup_table)

	return gamma_corrected_image

@st.cache_data
def checkHandOrientation(img, landmark1, landmark2):
	x1, y1 = img.shape[1]*landmark1.x, img.shape[0]*landmark1.y  
	x2, y2 = img.shape[1]*landmark2.x, img.shape[0]*landmark2.y

	center1 = img.shape[1]//2 
	center2 = 3*img.shape[0]//4 + 150

	#print(x1, y1)

	Rd1 = np.sqrt((x1-center2)**2 + (y1-center1)**2)
	Rd2 = np.sqrt((x2-center2)**2 + (y2-center1)**2)

	tan = math.atan2((y2 - center1), (x2 - center2))
	angle = round(math.degrees(tan))
	

	radius = 150
	a1, b1 = -int(radius*np.sqrt(2)/2) + center1, int(radius*np.sqrt(2)/2) + center2 
	a2, b2 = int(radius*np.sqrt(2)/2) + center1, -int(radius*np.sqrt(2)/2) + center2

	cv2.circle(img, (center2, center1), radius, (170, 110, 175), 2)
	cv2.line(img, (b1,a1), (b2,a2), (255,255,255), 2)

	if Rd1 < radius and Rd2 < radius:
		cv2.putText(img, f"Hand in circle: {'Good'}", (10, img.shape[1] - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 0), 2)
		hand_in_circle = True
	else:
		cv2.putText(img, f"Hand in circle: {'Bad'}", (10, img.shape[1] - 220), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 0, 0), 2)
		hand_in_circle = False

	cv2.putText(img, f"Hand orientation (in degrees): {angle}", (10, img.shape[1] - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 0), 2)
	return hand_in_circle, angle


@st.cache_data
def color_surface(img, w, h, color):

	(w1,w2) = w
	(h1,h2) = h

	img[w1:w2, h1:h2] = color
	#return img

@st.cache_data
def board(img, w, h):

	wn = lambda n: w//2**n
	hn = lambda n: h//2**n

	center = (7*wn(3)-22, 7*hn(3)-40)
	radius = 100

	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 0.95
	thickness = 2
	text = 'vs'
	ret, baseline = cv2.getTextSize(text, font, font_scale, thickness)
	text_width, text_height = ret
	text_x = int(round((w - 2*radius-80 - text_width) / 2))
	text_y = int(round((h - radius + text_height) / 2))

	#a1, b1 = -int(radius*np.sqrt(2)/2) + center[1]+15, int(radius*np.sqrt(2)/2) + center[0]-15
	#a2, b2 = int(radius*np.sqrt(2)/2) + center[1]-15, -int(radius*np.sqrt(2)/2) + center[0]+15

	colored = (240, 240, 240)
	# for angle in np.linspace(0, 2*np.pi):
	# 	y = int(radius*np.cos(angle)) + center[1] 
	# 	x = int(radius*np.sin(angle)) + center[0] 
	# 	if angle <= np.pi/2:
	# 		color_surface(img, (y, h), (x, w), colored)
	# 	elif angle > np.pi/2 and angle <= np.pi:
	# 		pass
	# 		color_surface(img, (h - 2*radius, y), (x, w), colored)
	# 	elif angle > np.pi and angle <= 1.5*np.pi:
	# 		color_surface(img, (h - 2*radius, y), (w-2*radius-5, x), colored)
	# 	else :
	# 		color_surface(img, (y, h), (w-2*radius-5, x), colored)

	color_surface(img, (0, h-3*radius-15), (w-2*radius-80, w), colored)
	#cv2.circle(img, center, radius, (120, 120, 120), 12)
	cv2.rectangle(img, (w - 2*radius-80, 0), (w - 2*radius-80, h-3*radius-15), (120,120,120), 5)
	cv2.rectangle(img, (w , h-3*radius-15), (w - 2*radius-80, h-3*radius-15), (120,120,120), 5)

	#cv2.arrowedLine(img, (b2, a2), (b1, a1), (255,255,255), 3, 8, 0, 0.1)
	cv2.rectangle(img, (0,0), (w, h), (120,120,120), 3)
	color_surface(img, (h-radius, h), (0, w-2*radius-80), colored)
	cv2.rectangle(img, (0,h), (w-2*radius-80, h-radius), (120,120,120), 3)

	cv2.putText(img, text, (text_x, text_y), font, font_scale, (255,0,0) , thickness)
	cv2.putText(img, "Human", (10, h-radius-5), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,255) , 1)
	cv2.putText(img, "AI Bot", (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255,255,0) , 1)
	
	return img


@st.cache_data
def gestures_recognizer(img, k=0):
	img.flags.writeable = False
	with GestureRecognizer.create_from_options(options) as recognizer:
		
		mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
		recognizer.recognize_async(mp_image, k)
		result = func.recognition

		k+=1

		
		img.flags.writeable = True
		if result.gestures:
			gestures = result.gestures[0][0]
			hand_in_circle, angle = checkHandOrientation(img, result.hand_landmarks[0][0], result.hand_landmarks[0][12])
			#title = f"{gestures.category_name} ({gestures.score:.2f})"

			hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
			hand_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(
				x=landmark.x, 
				y=landmark.y, 
				z=landmark.z) 
				for landmark in result.hand_landmarks[0]])

			mp_drawing.draw_landmarks(img, hand_landmarks_proto, mp_hands.HAND_CONNECTIONS, 
				mp_drawing_styles.get_default_hand_landmarks_style(),
				mp_drawing_styles.get_default_hand_connections_style())
			
		
			return img, gestures.category_name, gestures.score, hand_in_circle, angle
		else:
			return img, None, None, False, -np.float('inf')

@st.cache_data
def annotate(image, corresponding_images_path):
	'''
	This function will draw an appealing visualization of each fingers up of the both hands in the image.
	Args:
	image:The image of the hands on which the counted fingers are required to be visualized.
	Returns:
	output_image: A copy of the input image with the visualization of counted fingers.
	'''

	# Get the height and width of the input image.
	height, width, _ = image.shape

	radius = 100

	width_r = width - 2*radius 
	height_r = height - radius

	# Create a copy of the input image.
	output_image = image.copy()

	# Select the images of the hands prints that are required to be overlayed.
	########################################################################################################################

	# Iterate over the left and right hand.
	for hand_index, img_path in enumerate(corresponding_images_path):

		# Read the image including its alpha channel. The alpha channel (0-255) determine the level of visibility. 
		# In alpha channel, 0 represents the transparent area and 255 represents the visible area.


		hand_imageBGRA = cv2.resize(cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB), (125, 125))

		# Retrieve all the alpha channel values of the hand image. 
		##alpha_channel = hand_imageBGRA[:,:,-1]#

		# Retrieve all the blue, green, and red channels values of the hand image.
		# As we also need the three-channel version of the hand image. 
		hand_imageBGR = hand_imageBGRA[:,:,:3]

		# Retrieve the height and width of the hand image.
		hand_height, hand_width, _ = hand_imageBGR.shape


		


		ROI = hand_imageBGR.copy() #[alpha_channel==255]#[alpha_channel==255](hand_index * width_r//2)(hand_index * width_r//2) 
		# Update the ROI of the output image with resultant image pixel values after overlaying the handprint.
		output_image[230 - hand_index*200  : (height_r - 25) - hand_index*200  , 85 + width_r//12 : 85+(width_r//12 + hand_width)] = ROI #hand_imageBGR.copy()

		xG = ((300-hand_index*325) + hand_height)//2 + 50 
		yG = (-150 + width_r//12 + width_r//12 + hand_width) 

		names = os.path.splitext(img_path)[0].split('/')
		name1 = names[-1]
		name2 = names[-2] 

		height, width, _ = output_image.shape
		y_width = (width - (hand_height//2+15))//5

		center = (xG+280 + height)//2
		center1 = (center+height)//2


		#print(y_width+100, center)
		#print(y_width+50, center1)

		cv2.putText(output_image, f"{name1}", (yG, xG), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (hand_index*255, hand_index*255, ((hand_index+1)%2)*255), 2)
		#cv2.putText(output_image, f"{name2.replace('_', ' ')}", (y_width-10, xG + 280), cv2.FONT_HERSHEY_COMPLEX, 0.95, (0, 255, 255), 1)




	return output_image

@st.cache_data
def textMenu(img=None, text=None, color=None, height=None,  font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.95, thickness = 2):

	if height == None:
		w, height, _ = img.shape
	else:
		w, _, _ = img.shape

	w += 3*w//8 - 15

	ret, baseline = cv2.getTextSize(text, font, font_scale, thickness)
	text_width, text_height = ret
	text_x = int(round((w  - text_width) / 2))
	text_y = int(round((height + text_height)))

	cv2.putText(img, text, (text_x, text_y), font, font_scale, color , thickness)

	return img

@st.cache_data
def start_game(img):

	white = (255, 255, 255)
	w,h,_ = img.shape

	img[:, 25:h-25] = (25, 25, 25)

	img = textMenu(img=img, text='Rock-Paper-Scissors', height=10, color=(200, 125, 0))
	img = textMenu(img=img, text='The classic game of chance', height=45, color=white,
	 font_scale=0.475, thickness=1)
	img = textMenu(img=img, text='Welcome to RPS Game Vision', color=white,font_scale=0.775, thickness=2, height=90)
	img = textMenu(img, text='Are you strong enough to win this game?', color=white, font_scale=0.575, thickness=2, height=150)
	img = textMenu(img, text="Let's see that! How to play?", color=white, font_scale=0.575, thickness=2, height=190)
	img = textMenu(img, text="Sign to choose (see tutorial if you want) :", color=(0, 255, 0), font_scale=0.575, thickness=2, height=230)
	img = textMenu(img, text="Rock (hand closed), Paper (Open_palm), Scissors (2 fingers)", color=(0, 255, 0),
		font_scale=0.575, thickness=2, height=260)
	img = textMenu(img, text='Click Start to play', color=(155, 155, 205), font_scale=0.875, thickness=2, height=320)
	img = textMenu(img, text='Press Stop to exit', color=(155, 155, 205), font_scale=0.875, thickness=2, height=360)
	img = textMenu(img, text='Contact: lbtutorialcollege@gmail.com', color=(200, 125, 0), font_scale=0.775, thickness=2, height=420)
	return img

@st.cache_data
def end_game(img):

	white = (255, 255, 255)
	w,h,_ = img.shape

	img[:, 25:h-25] = (25, 25, 25)

	img = textMenu(img, text='Click Start to replay', color=(155, 155, 205), font_scale=0.875, thickness=2, height=320)
	img = textMenu(img, text='Press Stop to exit', color=(155, 155, 205), font_scale=0.875, thickness=2, height=360)
	img = textMenu(img, text='Contact: lbtutorialcollege@gmail.com', color=(200, 125, 0), font_scale=0.775, thickness=2, height=420)

	return img








