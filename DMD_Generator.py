#Lee Mask Algorithm and Controller for DMD

import sys
sys.path.insert(0,"/home/chris/Documents/Projects/DMDSLM/slmPy/") #not permanent
import argparse
import cv2
import numpy as np
import scipy.misc 
import slmpy
import math

slmFlag = False


# normalize image to range [0, 1]    
def normalize_image(image):
	img = cv2.normalize(image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	image8bit = np.round((2**8-1)*(img)).astype('uint8')
	return image8bit

def Lee_Mask(phaseMask,nu): #apply Lee's method to generate a DMD binary mask that corresponds to some input phasemask
	#phaseMask is the desired phase dist phi(x,y) in the IMAGE plane. 
	#nu is the central spatial frequency.
	#X, Y are some meshgrid object....
	f = (1.0/2)*(1+np.cos(2*np.pi*(X-Y)*nu-phaseMask)) #gotta carefully cast the type of phaseMask here.
	image8bit = normalize_image(f) #this is NOT binarized.
	return image8bit


def iterativeBinarize(input):

	output_array = np.zeros([ImgResY,ImgResX])

	for i in range(0,ImgResY):
		for j in range(0,ImgResX):
			if (input[i,j]<0.5):
				output_array[i,j] = 0.0
				print('test')
			else:
				output_array[i,j]=1.0

	return output_array

def twoPointOutput(a,x1,y1,x2,y2): #gaussian size parameters and some coordinates. 
	image = np.zeros([ImgResY, ImgResX])
	mask1 = np.exp(-a*((X-x1)**2+(Y-y1)**2)) #can I do this kind of calculation in the mesh grid?
	mask2 = np.exp(-a*((X-x2)**2+(Y-y2)**2))
	mask = mask1+mask2
	image = normalize_image(mask)
	#binned = np.around(image)
	#cv2.imshow('binned',binned)
	#print(binned)

	return image


def generateConstraintMatrix(target_amplitude): #just a quick and dirty method to make constrained regions. 
	
	image = int(round(target_amplitude)) #should round all values?
	return image


def iterative_Image_Plane():
	#implementation of the iterative method by Wu Cheng and Tao.
	cutoff = 40 #cutoff steps 
	#set input amplitude as uniform as well as some phase, gonna normalize each vector for now.
	a = twoPointOutput(.001,100,0,-100,0) #np.zeros([ImgResY, ImgResX])
	S = iterativeBinarize(a)

	p = np.ones([ImgResY, ImgResX])
	cv2.imshow('a',a)

	#set target values
	A_target = np.zeros([ImgResY, ImgResX])
	P_target = np.zeros([ImgResY, ImgResX])

	#generate constraint matrices
	#S = np.ones([ImgResY, ImgResX])
	S_flip = np.zeros([ImgResY, ImgResX])

	#fill the inverted constraint matrix
	for i in range(0,ImgResY):
		for j in range(0,ImgResX):
			if(S[i,j]==1):
				S_flip[i,j] = 0
			else:
				S_flip[i,j] = 1
	cv2.imshow('constraints_flip',S_flip)
	cv2.imshow('constraints',S)


	for n in range(0,cutoff): #the iterative mega-loop
		#calculate the complex amplitude
		U_calculated = np.fft.fft2(np.multiply(a,np.exp(1j*p))) #do a num fast fourier transform on the input plane
		U_calculated = np.fft.fftshift(U_calculated) #I think this is necessary

		A_calculated = np.absolute(U_calculated)
		P_calculated = np.angle(U_calculated)

		#Calculate ALPHA plane values
		A_alpha = np.multiply(A_target,S) + np.multiply(A_calculated,S_flip)
		P_alpha = np.multiply(P_target,S) + np.multiply(P_calculated,S_flip)
		U_alpha = np.multiply(A_alpha, np.exp(1j*P_alpha))

		#Calculate BETA plane values
		A_beta = np.multiply(A_target,S_flip) + np.multiply(A_calculated,S)
		P_beta = np.multiply(P_target,S_flip) + np.multiply(P_calculated,S)
		U_beta = np.multiply(A_beta, np.exp(1j*P_beta))

		#propagate our new complex fields backwards
		U_alpha_back = np.fft.ifft2(U_alpha)
		U_alpha_back = np.fft.ifftshift(U_alpha_back) #shift again
		P_alpha_back = np.angle(U_alpha_back)

		U_beta_back = np.fft.ifft2(U_beta)
		U_beta_back = np.fft.ifftshift(U_beta_back) #shift again

		P_beta_back = np.angle(U_beta_back)

		#generate the new phase array and iterate again
		p = np.angle((np.exp(1j*P_alpha_back)+np.exp(1j*P_beta_back)))

		n += 1 #increment 

	return p #return the final phase mask

if slmFlag == True:
    # create the object that handles the SLM array
    slm = slmpy.SLMdisplay(isImageLock = True)
    # retrieve SLM resolution (defined in monitor options)
    ImgResX, ImgResY = slm.getSize()
else:
    
    ImgResX = 792
    ImgResY = 600

ImgCenterX = ImgResX/2
ImgCenterY = ImgResY/2

x = np.linspace(0,ImgResX,ImgResX)
y = np.linspace(0,ImgResY,ImgResY)

# initialize image matrix
X, Y = np.meshgrid(x,y)

X = X - ImgCenterX
Y = Y - ImgCenterY

# generate circular window mask
maskRadius = 0
#I think this is just if there is some sort of circular window...
maskCircle = np.zeros((ImgResY, ImgResX), dtype = "uint8")
cv2.circle(maskCircle, (ImgCenterX, ImgCenterY), maskRadius, 255, -1)
maskCircle = normalize_image(maskCircle)



#this is just for testing
final_phase_mask = iterative_Image_Plane()
#this is what we're going to hit now
if slmFlag != True:
	cv2.imshow('phase hologram',final_phase_mask)
	cv2.waitKey()

