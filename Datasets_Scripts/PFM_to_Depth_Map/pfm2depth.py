import os
import sys
import numpy as np
import cv2 as cv
import re
import png
from glob import glob

def process_middlebury2021():
    scenes = glob('./dataset/Middlebury2021/*')
    for scene in scenes:
        process_pfm(scene)


def process_middlebury2014():
    scenes = glob('./dataset/Middlebury2014/*')
    for scene in scenes:
        process_pfm(scene)


#def process_pfm(scene):
def process_pfm(scene, path, i):
    """ Read Middlebury disparity map and calculate depth map.
    
        http://davis.lbl.gov/Manuals/NETPBM/doc/pfm.html
    """
    # ==> Read disparity map from *.pfm file
    #disp_path = f'{scene}/disp0GT.pfm'
    disp_path = f'{scene}/disp0.pfm'
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
    calib = f'{scene}/calib.txt'
    with open(calib, 'r') as f:
        lines = f.readlines()

    f = float(re.findall("\d+\.\d+", lines[0])[0])
    try:
        doffs = float(re.findall("\d+\.\d+", lines[2])[0])
    except:
        doffs = float(re.findall("\d+", lines[2])[0])
    baseline = float(re.findall("\d+\.\d+", lines[3])[0])
    
    # ==> Calculate depth map
    depth = baseline * f / (disp + doffs) # depth in [mm]

    # ==> Write depth map into 16-bit .png
    #depth_path = f'{scene}/img'+str(i)+".png"
    depth_path = path+'/img'+str(i)+".png"
    depth_uint16 = np.round(depth).astype(np.uint16)
    with open(depth_path, 'wb') as f:
        writer = png.Writer(width=width, height=height, bitdepth=16, greyscale=True)
        writer.write(f, np.reshape(depth_uint16, (-1, width)))

    return depth

def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    #print(str(len(sys.argv)))
    assert len(sys.argv) == 3, "Usage: python pfm2depth.py <path_to_pfm_disparity> <path_to_save_png_disparity>"
    scenes = glob(sys.argv[1]+'/*')
    #print(scenes)
    for i, scene in enumerate(scenes):
        print(str(i))
        print(str(scene))
        process_pfm(scene, sys.argv[2], i)

if __name__=='__main__':
    #process_middlebury2014()
    main()
    
