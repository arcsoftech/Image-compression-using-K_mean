"""
Author:Arihant Chhajed
Description: Machine Learning Assignment 3 - Image compression using Kmean cluastering
Class CS6375.002
"""
from skimage import io
import numpy as np
import os
import glob
import random
import matplotlib
import sys

"""
DEFAULT PARAMETERS
"""
image_path = glob.glob(os.path.join(os.getcwd(), "data", 'Koala.jpg')) #image to be compressed
K = 2 #no. of clusters
I = 100 #no. of times the K-mean should run

def init_centroids(X,K):
    """
    Initialize Centroid pixels of the clusters. A centroid pixed Id 3D vector for RGB channels.
    """
    c = random.sample(list(X),K)
    return c

def load_image(image_path):
    """
    Load image and convert image to 3D pixel array
    """
    image = io.imread(image_path)
    io.imshow(image)
    io.show()
    print("Size of the image is {} KB".format(round(os.path.getsize(image_path)/1024,2)))
    return image

def nearest_cluster(X,c):
    """
    Find nearest cluster for each pixel based ion euclidian distance
    """
    K = np.size(c,0)
    idx = np.zeros((np.size(X,0),1))
    arr = np.empty((np.size(X,0),1))
    for i in range(0,K):
        y = c[i]
        temp = np.ones((np.size(X,0),1))*y
        b = np.power(np.subtract(X,temp),2)
        a = np.sum(b,axis = 1)
        a.resize((np.size(X,0),1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr,0,axis=1)
    idx = np.argmin(arr, axis=1)
    return idx

def update_centroids(X,idx,K):
    """
    Update new centroids afters pixels are mapped to each cluster after every iteration
    """
    n = np.size(X,1)
    centroids = np.zeros((K,n))
    for i in range(0,K):
        ci = idx==i
        ci = ci.astype(int)
        total_number = sum(ci)
        ci.resize((np.size(X,0),1))
        total_matrix = np.matlib.repmat(ci,1,n)
        ci = np.transpose(ci)
        total = np.multiply(X,total_matrix)
        try:
            centroids[i] = (1/total_number)*np.sum(total,axis=0)
        except Exception:
            centroids[i] = 0                                                                                                                                                                                                                                                                                                                       
    return centroids

def kmean(X,initial_centroids,max_iters):
    """
    Kmean Algorithm for image clustering
    """
    m = np.size(X,0)
    K = np.size(initial_centroids,0)
    centroids = initial_centroids
    idx = np.zeros((m,1))
    for i in range(1,max_iters):
        idx = nearest_cluster(X,centroids)
        centroids = update_centroids(X,idx,K)
    return centroids,idx


if __name__ == "__main__":
    """
    CommandLine Input
    """
    if(sys.argv.__len__() > 1 and sys.argv.__len__() != 4):
        sys.exit("""
            IF you want default paramters then directly pass enter without any argument or Please provide the argument in the format as stated below:-
            python k-mean.py <Path_image_to_be_compressed> <K> <I>
            K: No. of clusters
            I: No. of iteration
            """)
    elif(sys.argv.__len__() > 1):
        try:
            image_path = glob.glob(sys.argv[1])
            K = int(sys.argv[2])
            I = int(sys.argv[3])

        except Exception as ex:
            if(type(ex).__name__ == "ValueError"):
                print("Please enter integer value for 'K' cluster  and 'I' Iterations or no of epoch")
            else:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print (message)

    else:
        print("Default paramters are taken")
    print("No of clusters is {} and no of iteration is {}".format(K,I))
    image = load_image(image_path[0])
    rows = image.shape[0]
    cols = image.shape[1]
    # image = image/255 # normalize the pixel
    X = image.reshape(image.shape[0]*image.shape[1],3) # representing pixel vector in RGB space
    initial_centroids=init_centroids(X,K)
    centroids,idx=kmean(X,initial_centroids,I)
    idx = nearest_cluster(X,centroids)
    X_recovered = centroids[idx]
    X_recovered = np.reshape(X_recovered, (rows, cols, 3))
    matplotlib.image.imsave('compressed.jpg', X_recovered)
    image_compressed = io.imread('compressed.jpg')
    io.imshow(image_compressed)
    io.show()
    print("Size of the image is {} KB".format(round(os.path.getsize('compressed.jpg')/1024,2)))
