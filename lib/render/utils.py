import os, sys, time, re, json
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
def imread8(im_file):
    ''' Read image as a 8-bit numpy array '''
    im = np.asarray(Image.open(im_file))
    return im

def read_png(res):
    import StringIO, PIL.Image
    img = PIL.Image.open(StringIO.StringIO(res)).convert('RGB')
    return np.asarray(img)

def read_mask(res):
    import StringIO, PIL.Image
    img = PIL.Image.open(StringIO.StringIO(res)).convert('RGB')
    # img = img.resize((640, 480))
    return np.asarray(img)


def read_npy(res):
    import StringIO
    return np.load(StringIO.StringIO(res))

# ---------------------------------------------------------------------------

# Image stuff
#RGB image of the camera view
def get_image(client):
    res = client.request('vget /camera/0/lit png')
    img = read_png(res)
    return img

#depth map
def get_depth(client):
    res = client.request('vget /camera/0/depth npy')
    depth = read_npy(res)
    return depth

#surface normal
def get_surface_normal(client):
    res = client.request('vget /camera/0/normal png')
    img = read_png(res)
    return img

#object mask
def get_object_mask(client):
    res = client.request('vget /camera/0/object_mask png')
    img = read_mask(res)
    return img

# ------------------------------------------------------------------------------
# Camera stuff

#camera location x,y,z
def get_camera_location(client):
    loc = client.request('vget /camera/0/location')
    return loc

#camera rotation
def get_camera_rotation(client):
    [pitch, yaw, roll] = client.request('vget /camera/0/rotation')
    return [roll, pitch, yaw]

#camera  x,y,z,rx,ry,rz ?(pitch, yaw, roll)
def get_camera_pose(client):
    [x, y, z, pitch, yaw, roll] = client.request('vget /camera/0/pose')
    return [x, y, z, roll, pitch, yaw]
# ------------------------------------

#set camera  x,y,z
#loc = [x,y,z]
def set_camera_location(client, loc):
    loc_string = " ".join(map(str, loc))
    res = client.request('vset /camera/0/location {}'.format(loc_string))
    return res


#set camera  rx,ry,rz
#rot = [rx,ry,rz]
def set_camera_rotation(client, rot):
    [roll, pitch, yaw] = rot
    rot = [pitch, yaw, roll]
    rot_string = " ".join(map(str, rot))
    res = client.request('vset /camera/0/rotation {}'.format(rot_string))
    return res

#set camera  x,y,z,rx,ry,rz
#pose = [x,y,z,rx,ry,rz]
def set_camera_pose_helper(client, pose):
    pose_string = " ".join(map(str, pose))
    res = client.request('vset /camera/0/pose {}'.format(pose_string))
    return res

def set_camera_pose(client, loc, rot):
    [roll, pitch, yaw] = rot
    rot = [pitch, yaw, roll]
    pose = loc + rot
    res = set_camera_pose_helper(client, pose)
    return res
# ---------------------------------------------------------------------------------------------------
# Object Stuff

## Get objects names from the scenes. Only cars, name the cars with "Vh" or "Vehicle" in the beginning
def get_objects(client):
    res = client.request('vget /objects')
    temp = res.split(' ')
    
    object_names = []
    
    for s in temp:
        if(s.startswith('Vh') or s.startswith('Vehicle') or s.startswith('Person') ):
            object_names.append(s)
    object_names = sorted(object_names)
    return object_names

#object location x,y,z
def get_object_location(client, obj_name):
    loc = client.request('vget /object/{}/location'.format(obj_name))
    return loc

#object rotation rx, ry, rz
def get_object_rotation(client, obj_name):
    [pitch, yaw, roll] = client.request('vget /object/{}/rotation'.format(obj_name))
    return [roll, pitch, yaw]

#object labeling color rgb
def get_object_color(client, obj_name):
    color = client.request('vget /object/{}/color'.format(obj_name))
    return color

# ------------------------------------
#object location x,y,z
def set_object_location(client, obj_name, loc):
    loc_string = " ".join(map(str, loc))
    res = client.request('vset /object/{}/location {}'.format(obj_name, loc_string))
    return res

#object rotation rx, ry, rz
def set_object_rotation(client, obj_name, rot):
    [roll, pitch, yaw] = rot
    rot = [pitch, yaw, roll]
    rot_string = " ".join(map(str, rot))
    res = client.request('vset /object/{}/rotation {}'.format(obj_name, rot_string))
    return res

#object labeling color rgb
# color = [r,g,b]
def set_object_color(client, obj_name, color):
    color_string = " ".join(map(str,color))
    res = client.request('vset /object/{}/color {}'.format(obj_name, color_string))
    return res

#hide object
def set_object_hide(client, obj_name):
    res = client.request('vset /object/{}/hide'.format(obj_name))
    return res


#show object
def set_object_show(client, obj_name):
    res = client.request('vset /object/{}/show'.format(obj_name))
    return res

# ---------------------------------------------------------------------------------------------------
#reset scene

def reset_scene(client, active_objects):
    res = client.request('vset /viewmode lit')

    for obj_name in active_objects:
        res = set_object_hide(client, obj_name)

    return


