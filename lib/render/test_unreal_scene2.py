
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter, OrderedDict

import numpy as np
from utils import *
from scipy import misc
import cv2
import math

# ----------------------------------------------------
from unrealcv import Client
# ----------------------------------------------------

# python test_unreal_scene1.py --port 9001
# car_model, color, x, y, theta
# x and y in -1 to 1, theta in 0 to 360

# 0: sedan, 1: suv, 2: truck, 3: person
# ('white', 'black', 'red', 'blue', 'brown', 'silver', 'cyan', 'yellow')

# DEFAULT_PARAM = '0.0;0.0;-0.49363598229599526;0.9014144508679249;320.90255860069726#0.0;1.0;-1.0;1.0;90.0'
DEFAULT_PARAM = '1.0;3.0;-0.11959371261162599;0.7107625177548988;97.82041735274184#0.0;1.0;-1.0;1.0;90.0'

# ----------------------------------------------------
def parse_object_parameters(s):

	object_class_dict = {0: 'sedan', 1: 'suv', 2: 'truck', 3: 'person'}
	color_class_dict = {0: 'white', 1: 'black', 2: 'red', 3: 'blue', 4: 'brown', 5: 'silver', 6: 'cyan', 7: 'yellow'}

	s = s.split(';')
	object_name = int(float(s[0]))
	color = int(float(s[1]))
	x = float(s[2])
	y = float(s[3])
	theta = float(s[4])

	object_name_string = object_class_dict[object_name]
	color_string = color_class_dict[color]

	dict ={"object_name":object_name_string, "color":color_string, "x":x, "y":y, "theta":theta}

	return dict
# ----------------------------------------------------

def parse_parameters(s):
	print(s)
	s = s.split('#')

	objects = []

	for object_parameter in s:
		objects.append(parse_object_parameters(object_parameter))

	return objects

# ----------------------------------------------------

def connect_client(port):
	CLIENT = Client(('localhost', port), None)
	CLIENT.connect()
	if not CLIENT.isconnected():
	    print('UnrealCV server is not running')
	    sys.exit(-1)

	res = CLIENT.request('vget /unrealcv/status')# The image resolution and port is configured in the config file.
	print(res)

	return CLIENT


# ----------------------------------------------------

parser = argparse.ArgumentParser()

#Unrealcv Setting
parser.add_argument('--port', default=9000, type=int)
parser.add_argument('--render_parameters', default=DEFAULT_PARAM)
parser.add_argument('--render_img_name', default='render.png')
parser.add_argument('--render_dir', default='render_test')

# ----------------------------------------------------

def main(args):
	CLIENT = connect_client(args.port)

	# --------------------------------------
	
	render_scene(CLIENT, args)

	return

# -----------------------------------------------------------
def render_scene(CLIENT, args):

	if (not os.path.exists(args.render_dir)):
		os.makedirs(args.render_dir)
	
	output_image=os.path.join(args.render_dir,args.render_img_name)
	CAMERA_LOCATION = [-4010.477539, -5317.555176, 2697.052246]
	CAMERA_ROTATION = [-0.313782, -22.125134, 37.976944]	

	res = set_object_color(CLIENT, 'SkySphere', [0,0,0])
	res = set_object_color(CLIENT, 'Plane1', [0,0,0])
	res = set_object_color(CLIENT, 'Plane2_13', [0,0,0])
	res = set_object_color(CLIENT, 'Plane3_2', [0,0,0])
	res = set_object_color(CLIENT, 'Plane4', [0,0,0])

	res = set_camera_pose(CLIENT, CAMERA_LOCATION, CAMERA_ROTATION)

	add_objects(CLIENT, args)
	get_objects(CLIENT)

	img = get_image(CLIENT) #save rgb

	misc.imsave(output_image.replace('.png','_view1.png'), img)
	# ---------------------------------------------------------------

	# ---------------------------------------------------------------
	#view 2 image
	CAMERA_LOCATION = [-2761.377441, 3690.708252, 3841.294922]
	CAMERA_ROTATION = [0, -42.800594, -55.718647]		
	res = set_camera_pose(CLIENT, CAMERA_LOCATION, CAMERA_ROTATION)
	get_objects(CLIENT); 
	img = get_image(CLIENT) #save rgb

	misc.imsave(output_image.replace('.png','_view2.png'), img)
	# ---------------------------------------------------------------
	#view 3 image
	CAMERA_LOCATION = [-5696.342773, -277.063965, 3695.119873]
	CAMERA_ROTATION = [0, -31.600555, -0.718658]		
	res = set_camera_pose(CLIENT, CAMERA_LOCATION, CAMERA_ROTATION)
	get_objects(CLIENT)
	img = get_image(CLIENT) #save rgb

	misc.imsave(output_image.replace('.png','_view3.png'), img)

	return 

# ----------------------------------------------------
def add_objects(CLIENT, args):

	render_parameters = parse_parameters(args.render_parameters)
	available_object_names = get_objects(CLIENT)
	res = reset_scene(CLIENT, available_object_names)

	# ----------------------------------------------------------------
	X_MIN = -2100; X_MAX = 3000
	Y_MIN = -4200.0; Y_MAX = 1000

	CAR_LEN = 400
	PERSON_LEN = 40

	x_scale = (X_MAX - X_MIN)/2.0; x_centre = (X_MAX + X_MIN)/2.0
	y_scale = (Y_MAX - Y_MIN)/2.0; y_centre = (Y_MAX + Y_MIN)/2.0
	# ----------------------------------------------------------------

	for param in render_parameters:
		object_name = param["object_name"]
		color = param["color"]

		x = param["x"]
		y = param["y"]
		theta = param["theta"]

		render_x = x*x_scale + x_centre
		render_y = y*y_scale + y_centre
		render_theta = theta

		# --------------------------
		if(object_name == "sedan"):
			render_theta = theta - 90
			render_id = "Vh_" + object_name + "_" + color

		if(object_name == "suv"):
			render_theta = theta + 90 #to reset at the canonical orientation
			render_id = "Vh_" + object_name + "_" + color

		if(object_name == "truck"):
			render_theta = theta - 90
			render_id = "Vh_" + object_name + "_" + color

		if(object_name == "person"):
			render_theta = theta - 90
			render_id = "Person_" + object_name + "_" + color
		# --------------------------

		object_name = "" 
		for available_object_name in available_object_names:
			if(available_object_name.startswith(render_id)):
				object_name = available_object_name
				break

		available_object_names.remove(object_name)
		add_object(CLIENT, object_name, render_x, render_y, render_theta)

	return 
# ----------------------------------------------------
def add_object(CLIENT, obj_name, x, y, theta=0):
	z = 0
	# print("adding {}".format(obj_name))
	res = set_object_location(CLIENT, obj_name, [x,y,z])
	res = set_object_rotation(CLIENT, obj_name, [0, 0, theta])

	res = set_object_show(CLIENT, obj_name)
	return
# ----------------------------------------------------

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)
