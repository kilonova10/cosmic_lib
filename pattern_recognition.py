import numpy as np
import matplotlib.pyplot as plt
import random

''' basic setup for 2D Gaussian dataset '''
class Class:
    def __init__(self, name,N, M, C, col):
        self.name = name
        self.count = N
        self.mean = M
        self.cov = C
        self.clusterx = []
        self.clustery = []
        self.colour = col
        ## to be filled onwards
        self.evalues = []
        self.evecs = []

from math import *

''' given a count, make a 2D Gaussian distribution with "count" points in it'''
def generate_gaussian(count):
    
    dimension = 2 #2D problems
    out = np.random.randn(count, dimension)

    return out

''' Generate and rotate an ellipse as per the equiprobability contour rules
the angle and axes lengths are determined by eigendecomposition outputs'''
def compute_contour(eigenvalues, eigenvectors, mean):
    max_ind = np.argmax(eigenvalues)
    min_ind = max(1-max_ind, 0)
    theta = np.arctan([eigenvectors.T[max_ind][1]/eigenvectors.T[max_ind][0]])

    a=np.sqrt(eigenvalues[max_ind])
    b=np.sqrt(eigenvalues[min_ind])
    
    t = np.linspace(0, 2*np.pi, 100)
    ell = np.array([a*np.cos(t) , b*np.sin(t)])  
    
    rotation = np.array([[cos(theta) , -sin(theta)],[sin(theta) , cos(theta)]])  
    #2-D rotation matrix

    r_ellipse = np.zeros((2,ell.shape[1]))
    for i in range(ell.shape[1]):
        r_ellipse[:,i] = np.dot(rotation,ell[:,i])
    
    cx = mean[0]+r_ellipse[0,:] 
    cy = mean[1]+r_ellipse[1,:]


    
    return (cx, cy)

''' use eigenvalues and eigenvectors to plot the 4 points around the equiprobability contour (works as a check)
for the result from the above function'''
def get_contour(eigenvalues, eigenvectors, mean):
    max_ind = np.argmax(eigenvalues)
    min_ind = max(1-max_ind, 0)
    points =[]
    
    
    p1 = mean.T-eigenvectors.T[max_ind]*np.sqrt(eigenvalues[max_ind])
    p2 = mean.T +eigenvectors.T[min_ind]*np.sqrt(eigenvalues[min_ind])
    p3 = mean.T - eigenvectors.T[min_ind]*np.sqrt(eigenvalues[min_ind])
    p4 = mean.T + eigenvectors.T[max_ind]*np.sqrt(eigenvalues[max_ind])
    points.append(p1)
    points.append(p2)
    points.append(p3)
    points.append(p4)
    return points

''' make the plot of data, ellipse and mean given a group of classes'''
def generate_plot(group):
        
    plt.figure(figsize = (20,10))
    for g in group:
        count = g.count
        mean= g.mean
        covariance = g.cov
        print(g.name)
        original = generate_gaussian(count)
        [eigenvalues, eigenvectors] = np.linalg.eig(covariance)
        g.evalues = eigenvalues
        g.evecs = eigenvectors 
        
        cx,cy = compute_contour(eigenvalues, eigenvectors, mean)
        g.ellipse = (cx,cy)
        
        points = get_contour(eigenvalues, eigenvectors, mean)
        g.ell_points = points
        
        # get magnitudes of shift of the points as l
        l = np.matrix(np.diag(np.sqrt(eigenvalues)))
        # the directions (eigenvectors) multiplied by the magnitudes will tell
        # every original point where to move next
        Q = np.matrix(eigenvectors) * l

        result = []
        x1_tweaked = []
        x2_tweaked = []
        # every point is transformed: shifted by the mean value, translated by
        # the sqrt of eigenvalues in the direction of the eigenvectors.
        # this follows the same rule as when determining the equiprobability contour points
        for i, j in original:
            value = np.array( [[i], [j]])
            transformed = (Q*value) + mean
            result.append(transformed)
            x1_tweaked.append(float(transformed[0]))
            x2_tweaked.append(float(transformed[1]))
    
        g.clusterx = x1_tweaked
        g.clustery = x2_tweaked
        
        plt.scatter(x1_tweaked, x2_tweaked, label = g.name, color =g.colour, s=10)
        plt.scatter(mean[0],mean[1],s=200, marker = "D", color = g.colour)
        for p in points:
            plt.scatter(p[0][0],p[0][1],s=50, marker = "D", color = g.colour)
        plt.plot(cx,cy, color = g.colour, linestyle = "--")
        plt.legend()
    plt.grid()
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Group Clusters', size = 20)
    plt.show()

 ''' MED CLASSIFIER ''' 

 ''' given two points, get their Euclidean distance'''
def get_dist(x1,x2,y1,y2):
    delx = (x2-x1)**2
    dely = (y2-y1)**2
    return np.sqrt(delx + dely)

''' given two means, classify each point inside the grid as "belonging" to either class 1 or class 2'''
def getMEDmap(mean1, mean2):
    m1 = mean1.flatten()
    m2 = mean2.flatten()
    xvals = np.arange(0,25, 1)
    yvals = np.arange(0,25, 1)
    
    MED = np.zeros((25, 25))
    i = 0
    for x in xvals:
        j = 0
        for y in yvals:
            dist1 = get_dist(x,m1[0], y, m1[1])
            dist2 = get_dist(x,m2[0],y,m2[1])
            MED[i,j] = dist1-dist2
            j+=1
        i+=1
    # pin the values to either 1 or 2 for separation
    MED[MED>0] = 1
    MED[MED<0] = 2
    return MED