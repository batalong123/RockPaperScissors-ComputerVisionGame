
import streamlit as st



def welcome():

	#title
	st.title("Rock Paper Scissors")
	st.subheader('The classic vision game of chance')

	#description
	text1 = """
	**Rock Paper Scissors** is a classic game that most of us have played at some point in our lives.
	It's a game in which two players make simultaneous moves by choosing one of the three options: 
	rock, paper, or scissors. The winner is determined by a set of rules: ***rock beats scissors, 
	scissors beat paper, and paper beats rock.***  

	Now, let's imagine you play against an AI agent that can play Rock Paper Scissors competitively.
	You have a privilege to play against this agent using a computer vision. Our agent  have the eyes
	to see your hand choice.

	**Are you enough strong to play rock-paper-scissor vision game?** Okay, I know, you are very strong for 
	that. Let's me show you a tutorial.
	"""

	st.markdown(text1, unsafe_allow_html=False)

	st.subheader('Tutorial: how to play rps vision game?')

	text2 = """
	The game is very simple:
	- *Put your hand such that webcam can detect it. When your hand is detected, a circle purple appears. 
	The white line indicates the orientation of your hand. It is very necessary for a gesture recognizer
	model. This model recognizes a rock sign, a paper sign and a scissors sign.*

	Now, the game have two boards:
	- *The bottom board shows you if your hand is in the cercle: green color indicates your hand is 
	in the cercle in contrary it is red. When a game begins, you have 15 seconds (time down) to response after this 
	time you lost a round.*
	- *The Right corner board shows you an information of game.*

	Okay, Let's go! Enjoy a computer vision game! 
	"""
	st.markdown(text2, unsafe_allow_html=False)
##*******************************************************************************************              




        



