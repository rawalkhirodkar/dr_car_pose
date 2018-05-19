
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


parser = argparse.ArgumentParser()

#Unrealcv Setting
parser.add_argument('--port', default=9000, type=int)

#Input options
parser.add_argument('--properties_json', default='unreal_data/properties.json')

#Settings for objects
parser.add_argument('--min_objects', default=1, type=int)
parser.add_argument('--max_objects', default=15, type=int)
parser.add_argument('--min_dist', default=350, type=int)
parser.add_argument('--margin', default=0.6, type=int)
parser.add_argument('--max_retries', default=50, type=int)
parser.add_argument('--max_exception_retries', default=50, type=int)

#Output options
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--num_images', default=5, type=int)
parser.add_argument('--filename_prefix', default='UNREAL_SCENE1')
parser.add_argument('--split', default='train')
parser.add_argument('--output_dir', default='render_data/render_output0')

# ----------------------------------------------------
def euclidean_dist(p1, p2):
	dx = p1[0] - p2[0];
	dy = p1[1] - p2[1]

	dist = np.sqrt(dx*dx + dy*dy)

	return dist

# ----------------------------------------------------
def min_dist(x1, y1, theta1, l1, x2, y2, theta2, l2):
	theta1 = np.radians(theta1)
	theta2 = np.radians(theta2)

	c1 = np.cos(theta1); s1 = np.sin(theta1)
	c2 = np.cos(theta2); s2 = np.sin(theta2)

	points1 = [];
	points1.append([x1, y1])
	points1.append([x1 + l1*c1, y1 + l1*s1])
	points1.append([x1 - l1*c1, y1 - l1*s1])

	points2 = [];
	points2.append([x2, y2])
	points2.append([x2 + l2*c2, y2 + l2*s2])
	points2.append([x2 - l2*c2, y2 - l2*s2])

	min_dist = float('inf')
	for p1 in points1:
		for p2 in points2:
			dist = euclidean_dist(p1, p2)
			if(dist < min_dist):
				min_dist = dist

	return min_dist
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

def main(args):
	# --------------------------------------
	CLIENT = connect_client(args.port)

	# --------------------------------------
	num_digits = 6
	prefix = '%s_%s_' % (args.filename_prefix, args.split)
	img_template = '%s%%0%dd.png' % (prefix, num_digits)
	scene_template = '%s%%0%dd.json' % (prefix, num_digits)
	seg_template = '%s%%0%dd.png' % (prefix, num_digits)
	depth_template = '%s%%0%dd.png' % (prefix, num_digits)
	normal_template = '%s%%0%dd.png' % (prefix, num_digits)

	output_image_dir = os.path.join(args.output_dir, args.split, "images")
	output_scene_dir = os.path.join(args.output_dir, args.split, "scenes")
	output_all_seg_dir = os.path.join(args.output_dir, args.split, "segmentation", "all")
	output_individual_seg_dir = os.path.join(args.output_dir, args.split, "segmentation", "individual")
	output_scene_file = os.path.join(args.output_dir, args.split, args.filename_prefix+"_scenes.json")
	output_depth_dir = os.path.join(args.output_dir, args.split, "depths")
	output_normal_dir = os.path.join(args.output_dir, args.split, "normals")

	img_template = os.path.join(output_image_dir, img_template)
	scene_template = os.path.join(output_scene_dir, scene_template)
	seg_template = os.path.join(output_all_seg_dir, seg_template)
	depth_template = os.path.join(output_depth_dir, depth_template)
	normal_template = os.path.join(output_normal_dir, normal_template)

	if not os.path.isdir(output_image_dir):
		os.makedirs(output_image_dir)
	if not os.path.isdir(output_scene_dir):
		os.makedirs(output_scene_dir)
	if not os.path.isdir(output_all_seg_dir):
		os.makedirs(output_all_seg_dir)
	if not os.path.isdir(output_individual_seg_dir):
		os.makedirs(output_individual_seg_dir)
	if not os.path.isdir(output_depth_dir):
		os.makedirs(output_depth_dir)
	if not os.path.isdir(output_normal_dir):
		os.makedirs(output_normal_dir)
	
	all_scene_paths = []
	active_objects = get_objects(CLIENT) #all initially

	res = reset_scene(CLIENT, active_objects)

	for i in range(args.num_images):
		img_path = img_template % (i + args.start_idx)
		scene_path = scene_template % (i + args.start_idx)
		seg_path = seg_template % (i + args.start_idx)
		depth_path = depth_template % (i + args.start_idx)
		normal_path = normal_template % (i + args.start_idx)

		all_scene_paths.append(scene_path)
		num_objects = random.randint(args.min_objects, args.max_objects)

		exception_count = 0
		while (True):
			try:
				render_scene(CLIENT, args,
					num_objects=num_objects,
					output_index=(i + args.start_idx),
					output_split=args.split,
					output_image=img_path,
					output_scene=scene_path,
					output_seg=seg_path,
					output_depth=depth_path,
					output_normal=normal_path
				)
				break
			except Exception as e:
				exception_count += 1
				print(e)

				if(exception_count >= args.max_exception_retries):
					exception_count = 0
					CLIENT.disconnect()
					CLIENT = connect_client(args.port)
		
	
	# After rendering all images, combine the JSON files for each scene into a
	# single JSON file.
	all_scenes = []
	for scene_path in all_scene_paths:
		with open(scene_path, 'r') as f:
			all_scenes.append(json.load(f))
	
	output = {
		'info': {
			'split': args.split,
		},
		'scenes': all_scenes
	}
	with open(output_scene_file, 'w') as f:
		json.dump(output, f, indent=2)


	return
# -----------------------------------------------------------
def add_noise(img, color_mask, noise_sigma=10):

	img_copy = np.copy(img)
	# img_copy = img_copy.astype(np.float)


	# # noise = noise_sigma*np.random.randn(img.shape[0], img.shape[1], img.shape[2])
	# noise = noise_sigma*np.random.uniform(0, 1, (img.shape[0], img.shape[1], img.shape[2]))
	indices = np.where(np.all(color_mask > [0, 0, 0], axis=-1)) #syn pixels
	# img_copy[indices[0], indices[1], :] += noise[indices[0], indices[1], :]
	
	# img_copy = img_copy.astype(np.uint8)

	blur = cv2.blur(img,(30, 30))
	img_copy[indices[0], indices[1], :] = blur[indices[0], indices[1], :]
	return img



# -----------------------------------------------------------
def render_scene(CLIENT, args,
	num_objects=5,
	output_index=0,
	output_split='none',
	output_image='render_output/render.png',
	output_scene='render_output/render_json',
	output_seg='render_output/render_seg.png',
	output_depth='render_output/render_depth.png',
	output_normal='render_output/render_normal.png'
	):
	
	print("{} num_objects:{}".format(output_image, num_objects))

	CAMERA_LOCATION = [-2583.283203, -5715.984863, 2749.33252]
	CAMERA_ROTATION = [9.026343, -29.059454, 55.230694]		

	res = set_object_color(CLIENT, 'SkySphere', [0,0,0])
	res = set_object_color(CLIENT, 'Plane1', [0,0,0])
	res = set_object_color(CLIENT, 'Plane2_13', [0,0,0])

	res = set_camera_pose(CLIENT, CAMERA_LOCATION, CAMERA_ROTATION)

	# This will give ground-truth information about the scene and its objects
	scene_struct = {
		'split': output_split,
		'image_index': output_index,
		'image_filename': os.path.basename(output_image),
		'objects': [],
		'directions': {},
	}

	active_objects, unreal_object_info = add_random_objects(CLIENT, scene_struct, num_objects, args)

	img = get_image(CLIENT) #save rgb
	seg_img = get_object_mask(CLIENT) #save semseg
	depth_img = get_depth(CLIENT) #save npy
	normal_img = get_surface_normal(CLIENT) #save surface normal

	img = add_noise(img, seg_img)

	misc.imsave(output_image, img)
	misc.imsave(output_seg, seg_img)
	misc.imsave(output_depth, depth_img)
	misc.imsave(output_normal, normal_img)

	save_object_segmentations(CLIENT, active_objects, output_seg.replace("all","individual")[:-4])

	scene_struct['objects'] = unreal_object_info
	with open(output_scene, 'w') as f:
		json.dump(scene_struct, f, indent=2)

	return 
# ----------------------------------------------------
#this will make all the objects hide at the end 
def save_object_segmentations(CLIENT, active_objects, image_id):

	for object_name in active_objects:
		res = set_object_hide(CLIENT, object_name)

	for object_id, object_name in enumerate(active_objects):
		image_path = image_id + "_" + str(object_id) + ".png"
		res = set_object_show(CLIENT, object_name)
		img = get_object_mask(CLIENT) #save semseg	
		misc.imsave(image_path, img)
		res = set_object_hide(CLIENT, object_name)

	res = CLIENT.request('vset /viewmode lit')
	return


# ----------------------------------------------------
def add_random_objects(CLIENT, scene_struct, num_objects, args):
	all_object_names = get_objects(CLIENT)
	total_objects = len(all_object_names)

	positions = []
	active_objects = [] #list of object names
	unreal_object_info = []

	# available_object_ids = list(range(total_objects))
	available_object_names = all_object_names

	X_MIN = -1600.0; X_MAX = 1300.0
	# Y_MIN = -2700.0; Y_MAX = 2000.0
	Y_MIN = -3500.0; Y_MAX = 2800.0

	CAR_LEN = 350
	PERSON_LEN = 40

	x_scale = (X_MAX - X_MIN)/2.0; x_centre = (X_MAX + X_MIN)/2.0
	y_scale = (Y_MAX - Y_MIN)/2.0; y_centre = (Y_MAX + Y_MIN)/2.0


	object_categories = ['Vh', 'Vh', 'Vh', 'Person'] #3:1 ratio

	for i in range(num_objects):
		unreal_object_name = ""
		
		# ----------choose an object---------------
		while(True):
			object_type = random.choice(object_categories) #'Vh' or 'Person'
			random.shuffle(available_object_names) #keep shuffling
			for name in available_object_names:
				if(name.startswith(object_type)):
					unreal_object_name = name
					break
			if(unreal_object_name != ""):
				break

		# -------------------------------------------
		#obj_name = Vh_<car_type>_<color>_<id>
		#obj_name = Person_person_<color>_<id>
		temp = unreal_object_name.split("_") 
		object_type = temp[0] #Vh or Person
		object_name = temp[1]; color = temp[2]

		available_object_names.remove(unreal_object_name)
		num_tries = 0
		# -------only render-----------------

		#---------choose x and y ------------------------
		while True:
			num_tries += 1
			if(num_tries > args.max_retries):
				res = reset_scene(CLIENT, active_objects)
				return add_random_objects(CLIENT, scene_struct, num_objects, args)

			x = random.uniform(-1, 1)
			y = random.uniform(-1, 1)
			theta = 360.0 * random.random()

			render_x = x*x_scale + x_centre
			render_y = y*y_scale + y_centre
			render_theta = theta

			dists_good = True

			# ----------------------------------------------	
			object_length = CAR_LEN

			if(object_name == "sedan"):
				render_theta = theta - 90

			if(object_name == "suv"):
				render_theta = theta + 90 #to reset at the canonical orientation

			if(object_name == "truck"):
				render_theta = theta - 90

			if(object_name == "person"):
				render_theta = theta - 90
				object_length = PERSON_LEN

			# ----------------------------------------------

			for (render_x1, render_y1, render_theta1, object_length1) in positions:
				dist = min_dist(render_x, render_y, render_theta, object_length, render_x1, render_y1, render_theta1, object_length1)
				if(dist < args.min_dist):
					dists_good = False
					break

			if dists_good:
				break
		
		add_object(CLIENT, unreal_object_name, render_x, render_y, render_theta)
		positions.append((render_x, render_y, theta, object_length))	

		active_objects.append(unreal_object_name)
		unreal_object_info.append({
			'object_type': object_type,
			'object_name': object_name,
			'color': color,
			'instance_id': i,
			'3d_coords': (x,y,0),
			'rotation': theta,
			'render_3d_coords': (render_x, render_y, 0)
		})

	#need to check the min pixel per object constraint as well!

	return active_objects, unreal_object_info
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
