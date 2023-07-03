import streamlit as st
from PIL import Image
from module.contact import contact_me
from module.greetings import welcome
from module.rps_game import ComputerVisionGame
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration


#to get connection
st.set_page_config(
page_title="Rock-Paper-Scissors",
page_icon= ":smiley:",
layout="centered",
initial_sidebar_state="expanded")

file = 'logo/logo.png'
image = Image.open(file)
img= st.sidebar.image(image, use_column_width=True)

st.sidebar.title('Section')
page_name = st.sidebar.selectbox('Select page:', ("Welcome", "Play vision game","Contact"))

RTC_CONFIGURATION = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})



if page_name == "Welcome":
	welcome()

if page_name == "Play vision game":
	st.title('RPS game')
	st.write('Computer vision')
	ctx = webrtc_streamer(key='OpenCV_WebRTC', mode=WebRtcMode.SENDRECV, 
		rtc_configuration=RTC_CONFIGURATION,
		media_stream_constraints = {'video':True, "audio":False},
		video_processor_factory=ComputerVisionGame, async_processing=True)
	if ctx.video_transformer:
		ctx.video_transformer.game_running = st.button('Start game')

if page_name == "Contact":
	contact_me()
