import os
import sys
import numpy as np
import cv2 as cv
import re
import png
from glob import glob

def sorting_data_sintel(scene, path, i):
    scenes = sorted(glob(scene+'/*.png'))
    #print(scenes)
    #sorted(scenes.glob('*.png'))
    index = int(i)
    for j, im in enumerate(scenes):
        #print(str(i))
        #print(str(im))
        img = cv.imread(im, cv.IMREAD_UNCHANGED)
        if i != 0 :
            index = int(index)+1
            cv.imwrite(path+'/'+str(index)+'.png',img)
            #index = int((str(i)+str(j)))
        else :
            cv.imwrite(path+'/'+str(j)+'.png',img)
            index = int(j)
    return index
        
def sintel():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    #print(str(len(sys.argv)))
    assert len(sys.argv) == 3, "Usage: python data_script.py <path_dataset> <path_final_dataset>"
    scenes = sorted(glob(sys.argv[1]+'/*'))
    #print(scenes)
    #sorted(scenes.glob('*'))
    #print(str(scenes))
    index = 0
    for i, scene in enumerate(scenes):
        #print(str(i))
        #print(str(scene))
        if index != 0:
            index = sorting_data_sintel(scene, sys.argv[2], index)
        else:
            index = sorting_data_sintel(scene, sys.argv[2], i)

def sorting_data_instereo2k(scene, path_left, path_right, path_disp, i):
    scenes = sorted(glob(scene+'/*.png'))
    #print(scenes)
    #sorted(scenes.glob('*.png'))
    img_l = cv.imread(scenes[0], cv.IMREAD_UNCHANGED)
    img_r = cv.imread(scenes[2], cv.IMREAD_UNCHANGED)
    img_d = cv.imread(scenes[1], cv.IMREAD_UNCHANGED)
    cv.imwrite(path_left+'/'+str(i)+'.png',img_l)
    cv.imwrite(path_right+'/'+str(i)+'.png',img_r)
    cv.imwrite(path_disp+'/'+str(i)+'.png',img_d)

def instereo2k():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    #print(str(len(sys.argv)))
    assert len(sys.argv) == 5, "Usage: python data_script.py <path_dataset> <path_final_dataset_left> <path_right> <path_disp>"
    
    scenes = sorted(glob(sys.argv[1]+'/*'))
    #print(scenes)
    #sorted(scenes.glob('*'))
    #print(str(scenes))
    index = 0
    for i, scene in enumerate(scenes):
        #print(str(i))
        #print(str(scene))
        sorting_data_instereo2k(scene, sys.argv[2],sys.argv[3], sys.argv[4], i)

def sorting_data_deth(scene, path_left, path_right, i):
    scenes = sorted(glob(scene+'/*.png'))
    #print(scenes)
    #sorted(scenes.glob('*.png'))
    img_l = cv.imread(scenes[0], cv.IMREAD_UNCHANGED)
    img_r = cv.imread(scenes[1], cv.IMREAD_UNCHANGED)
    cv.imwrite(path_left+'/'+str(i)+'.png',img_l)
    cv.imwrite(path_right+'/'+str(i)+'.png',img_r)

def deth_data():
    assert len(sys.argv) == 4, "Usage: python data_script.py <path_dataset> <path_final_dataset_left> <path_right>"
    scenes = sorted(glob(sys.argv[1]+'/*'))
    #print(scenes)
    #sorted(scenes.glob('*'))
    #print(str(scenes))
    index = 0
    for i, scene in enumerate(scenes):
        #print(str(i))
        #print(str(scene))
        sorting_data_deth(scene, sys.argv[2],sys.argv[3], i)

def process_pfm(disp_path, calib, path, i):
    """ Read Middlebury disparity map and calculate depth map.
    
        http://davis.lbl.gov/Manuals/NETPBM/doc/pfm.html
    """
    # ==> Read disparity map from *.pfm file
    #disp_path = f'{scene}/disp0GT.pfm'
    #disp_path = f'{scene}/disp0.pfm'
    disp = cv.imread(disp_path, cv.IMREAD_UNCHANGED)
    with open(disp_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())    # read disparity scale factor
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

    disp = disp * scale

    # ==> Read calibration file
    #calib = f'{scene}/calib.txt'
    with open(calib, 'r') as f:
        lines = f.readlines()

    f = float(re.findall("\d+\.\d+", lines[0])[0])
    try:
        doffs = float(re.findall("\d+\.\d+", lines[2])[0])
    except:
        doffs = float(re.findall("\d+", lines[2])[0])
    try:
        baseline = float(re.findall("\d+\.\d+", lines[3])[0])
    except:
        baseline = float(re.findall("\d+", lines[3])[0])
    
    # ==> Calculate depth map
    depth = baseline * f / (disp + doffs) # depth in [mm]

    # ==> Write depth map into 16-bit .png
    #depth_path = f'{scene}/img'+str(i)+".png"
    depth_path = path+'/'+str(i)+".png"
    depth_uint16 = np.round(depth).astype(np.uint16)
    with open(depth_path, 'wb') as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=True)
        writer.write(f, np.reshape(depth_uint16, (-1, width)))

    return depth

def sorting_carla_res(scene, path_left, path_right, path_disp, i):
    scenes = sorted(glob(scene+'/*'))
    print(scenes)
    #sorted(scenes.glob('*.png'))
    img_l = cv.imread(scenes[3], cv.IMREAD_UNCHANGED)
    img_r = cv.imread(scenes[4], cv.IMREAD_UNCHANGED)
    cv.imwrite(path_left+'/'+str(i)+'.png',img_l)
    cv.imwrite(path_right+'/'+str(i)+'.png',img_r)
    process_pfm(scenes[1], scenes[0], path_disp, i)

def carla_high_res():
    assert len(sys.argv) == 5, "Usage: python data_script.py <path_dataset> <path_final_dataset_left> <path_right> <path_disp>"
    scenes = sorted(glob(sys.argv[1]+'/*'))
    #print(scenes)
    #sorted(scenes.glob('*'))
    #print(str(scenes))
    index = 0
    for i, scene in enumerate(scenes):
        #print(str(i))
        #print(str(scene))
        sorting_carla_res(scene, sys.argv[2],sys.argv[3], sys.argv[4], i)

def kitti2015():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    #print(str(len(sys.argv)))
    assert len(sys.argv) == 6, "Usage: python data_script.py <path_dataset> <path_final_dataset_left> <path_right> <path_disp> <train>"
    
    l_path =  sys.argv[1] + '/' + 'image_2'
    l_images = sorted(glob(l_path+'/*_10.png'))
    r_path =  sys.argv[1] + '/' + 'image_3'
    r_images = sorted(glob(r_path+'/*_10.png'))
    if sys.argv[5] == 'True' :
        d_path =  sys.argv[1] + '/' + 'disp_occ_0'
        d_images = sorted(glob(d_path+'/*.png'))
    #print(scenes)
    #sorted(scenes.glob('*'))
    #print(str(scenes))
    #print(len(l_images))
    for i in range(len(l_images)):
        #print(i)
        img_l = cv.imread(l_images[i], cv.IMREAD_UNCHANGED)
        img_r = cv.imread(r_images[i], cv.IMREAD_UNCHANGED)
        cv.imwrite(sys.argv[2]+'/'+str(i)+'.png',img_l)
        cv.imwrite(sys.argv[3]+'/'+str(i)+'.png',img_r)
        if sys.argv[5] == 'True' :
            img_d = cv.imread(d_images[i], cv.IMREAD_UNCHANGED)
            cv.imwrite(sys.argv[4]+'/'+str(i)+'.png',img_d)
            
def kitti2012():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    #print(str(len(sys.argv)))
    assert len(sys.argv) == 6, "Usage: python data_script.py <path_dataset> <path_final_dataset_left> <path_right> <path_disp> <train>"
    
    l_path =  sys.argv[1] + '/' + 'image_0'
    l_images = sorted(glob(l_path+'/*_10.png'))
    r_path =  sys.argv[1] + '/' + 'image_1'
    r_images = sorted(glob(r_path+'/*_10.png'))
    if sys.argv[5] == 'True' :
        d_path =  sys.argv[1] + '/' + 'disp_occ'
        d_images = sorted(glob(d_path+'/*.png'))
    #print(scenes)
    #sorted(scenes.glob('*'))
    #print(str(scenes))
    #print(len(l_images))
    for i in range(len(l_images)):
        img_l = cv.imread(l_images[i], cv.IMREAD_UNCHANGED)
        img_r = cv.imread(r_images[i], cv.IMREAD_UNCHANGED)
        cv.imwrite(sys.argv[2]+'/'+str(i)+'.png',img_l)
        cv.imwrite(sys.argv[3]+'/'+str(i)+'.png',img_r)
        if sys.argv[5] == 'True' :
            img_d = cv.imread(d_images[i], cv.IMREAD_UNCHANGED)
            cv.imwrite(sys.argv[4]+'/'+str(i)+'.png',img_d)


if __name__=='__main__':
    #sintel()
    #instereo2k()
    #deth_data()
    #carla_high_res()
    #kitti2015()
    kitti2012()