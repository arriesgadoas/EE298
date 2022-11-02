#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:57:37 2022

@author: asarriesgado@up.edu.ph

Instructions:
    
    1. in terminal run $python <filename-of-this-python-script> <source_image>
        ex: python arriesgado_HW1_EE298_MML.py myImage.jpg
        
    2. select four corners of distorted rectangle
    
    3. close window
    
    4. new window will open showing the transformed image
    
    5. output image is saved as "output.jpg"

"""
import numpy as np
from matplotlib import pyplot as plt
import math
import matplotlib.image as mpimg
import sys

point_ctr = 0
pts = []

def get_points(event): 
    global point_ctr
    # checking for left mouse clicks
    if point_ctr < 4:
        plt.scatter(event.xdata, event.ydata)
        fig.canvas.draw()
        pts.append([event.xdata,event.ydata])
        point_ctr += 1
        print(pts)
        if point_ctr == 4:
            plt.pause(0.5)
            plt.close()
        
def get_projection_matrix(a):
    
    #a = np.array([[279,204],[277,422],[402,461],[397,274]])
    print("a:")
    print(a)
    L2_norms = np.sum(np.abs(a)**2,axis=-1)**(1/2)
    ref_index = list(L2_norms).index(np.amin(L2_norms))
    ref_point = a[ref_index]

    #shift other points w.r.t. reference point
    a_shifted = a-ref_point  #a_shifted is the transformed coordinates from the "ground truth" rectangle
    #print(a_shifted)
    x_shifted,y_shifted = a_shifted.T

    #remove ref_point before sorting remaining points
    a_no_ref = np.delete(a_shifted,ref_index,axis=0)
    #a_no_ref = np.delete(a,ref_index,axis=0)

    #create reference axis for angle measurement
    r = np.array([-1,0])
    r_norm = np.linalg.norm(r)

    #create empty list for angles
    theta = []

    for i in range(3):
        #get angle between vector and reference axis
        angle = math.acos(np.dot(a_no_ref[i],r)/(np.linalg.norm(a_no_ref[i])*r_norm))
        #convert to degree
        angle = math.degrees(angle)
        #append value to theta list
        theta.append(angle)

    #stack coordinates with their respective theta
    a_no_ref = np.delete(a,ref_index,axis=0)
    xx = np.vstack((a_no_ref.T, theta))
    # print(xx)

    #get sorted indices of theta
    i_arr = np.argsort(theta)
    # print(i_arr)

    #using the indices order of theta, arrange the x,y coordinates and theta
    p1 = np.take_along_axis(xx[0],i_arr,axis=0)
    p2 = np.take_along_axis(xx[1],i_arr,axis=0)
    p3 = np.take_along_axis(xx[2],i_arr,axis=0)

    #stack sorted array and transpose before stacking the origin on top
    xx = np.vstack((p1,p2,p3)).T

    #attach angle to origin (which is theta = 0)
    origin = a[ref_index].tolist()
    #origin = a_shifted[ref_index].tolist()
    origin.append(0.)

    #stack origin on top of other points
    xx = np.vstack((origin,xx))

    #delete angle column, done sorting no need for it
    a_sorted = np.delete(xx,2,axis=1).T
    # print(a_sorted.T)

    #create the homogeneous coordinates of the distorted rectange
    a_homo = np.vstack((a_sorted,np.ones(4))).T
    print("a_homo:")
    print(a_homo)
    
    #****** identification of dimenstions of ground truth rectangle *****#
    p1 = a_sorted.T[0]
    p2 = a_sorted.T[1]
    p3 = a_sorted.T[2]
    p4 = a_sorted.T[3]

    width = []
    v1 = (p4-p1)[0]
    width.append(v1)

    v2 = (p3-p2)[0]
    width.append(v2)

    # print(width)
    N = np.amax(width)
    print("N:")
    print(N)

    height = []
    v1 = (p2-p1)[1]
    height.append(v1)

    v2 = (p4-p3)[1]
    height.append(v2)

    # print(height)
    M = np.amax(height)
    print("M:")
    print(M)

    b = np.array([[a_homo[0][0],a_homo[0][1]],[a_homo[0][0],a_homo[0][1]+M],[a_homo[0][0]+N,a_homo[0][1]+M],[a_homo[0][0]+N,a_homo[0][1]]])

    #create homogenous coordinates of the "ground truth" rectangle
    b_homo = np.vstack((b.T,np.ones(4))).T

    # print(a_homo)
    print("b_homo:")
    print(b_homo)
   
    #create A matrix, please refer to the documentation for this part
    A = np.zeros([8,9])

    for i in range(4):
        A[2*i] = np.array([a_homo[i][0], a_homo[i][1], 1, 0, 0, 0, -b_homo[i][0]*a_homo[i][0], -b_homo[i][0]*a_homo[i][1], -b_homo[i][0]])
        A[(2*i)+1] = np.array([0, 0, 0, a_homo[i][0], a_homo[i][1], 1, -b_homo[i][1]*a_homo[i][0], -b_homo[i][1]*a_homo[i][1], -b_homo[i][1]])
    # print(A)

    U,S,V = np.linalg.svd(A)

    h = V[8]/V[8][8]

    H = np.reshape(h,[3,3])
    print("H:")
    print(H.round(4))
    print("a_homo.T")
    print(a_homo.T)
    fixed = np.matmul(H,a_homo.T)#H@a_homo.T
    print("fixed:")

    print(fixed)
    fixed = (fixed/fixed[2])
    print("fixed_2:")
    print(fixed)
   
    return H

def transform(a,t):
    warp_coords = np.matmul(np.linalg.inv(t),a)#(np.linalg.inv(t)@a)
    print(np.amin(warp_coords[2]))
    warp_coords = (warp_coords/warp_coords[2]).round().astype(int)
    # print(np.amin(warp_coords[2]))
    return warp_coords

if __name__=="__main__":
    
    args = sys.argv[1:]
    img = mpimg.imread(args[0])
    fig_ = plt.imshow(img)
    fig = fig_.get_figure()
    cid = fig.canvas.mpl_connect('button_press_event', get_points) 
    plt.show()
    
    data = np.array(pts)
    
    x,y = data.T
    
    #define transformation matrix
    t = get_projection_matrix(data)
    
    #get dimensions of image
    M, N = img.shape[:2]
    
    #create  homogenous coordinates 
    simple_coords = np.indices((N, M)).reshape(2, -1)
    homo_coords = np.vstack((simple_coords, np.ones(simple_coords.shape[1]))).astype(int)
    print(homo_coords)
    
    #get orig coordinates
    x_orig,y_orig = homo_coords[0],homo_coords[1]
    
    #apply transformation
    transformed_coords = transform(homo_coords,t)
    
    #get new coordinates
    x_trans, y_trans = transformed_coords[0], transformed_coords[1]
    
    #get indices of new image inside the original dimensions
    indices = np.where((x_trans >= 0) & (x_trans < N) & (y_trans >= 0) & (y_trans < M))   
    xpix_orig, ypix_orig = x_orig[indices], y_orig[indices]
    xpix_trans, ypix_trans = x_trans[indices], y_trans[indices]
    
    #create blank image with dimensions same as orig image
    img_trans = np.zeros_like(img)
    
    #assign pixel values to blank image to see transformed image
    img_trans[ypix_orig,xpix_orig] = img[ypix_trans, xpix_trans]
    
    #display image
    plt.imshow(img_trans)
    plt.savefig("output.jpg")
    plt.show()
    
    




