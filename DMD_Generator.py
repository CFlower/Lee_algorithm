#Lee Mask Algorithm and Controller for DMD

import sys
sys.path.insert(0,"/home/chris/Documents/Projects/DMDSLM/slmPy/") #not permanent
import argparse
import cv2
import numpy as np
import scipy.misc 
import slmpy
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

slmFlag = False

def Lee_Mask(phaseMask,nu): #apply Lee's method to generate a DMD binary mask that corresponds to some input phasemask
	#phaseMask is the desired phase dist phi(x,y) in the IMAGE plane. 
	#nu is the central spatial frequency.
	#X, Y are some meshgrid object....
	f = (1.0/2)*(1+np.cos(2*np.pi*(X-Y)*nu-phaseMask)) #gotta carefully cast the type of phaseMask here.
	#image8bit = normalize_image(f) #this is NOT binarized.
	return f

def iterativeBinarize(input, cutoff_frac):

	output_array = np.zeros([ImgResY,ImgResX])

	for i in range(0,ImgResY):
		for j in range(0,ImgResX):
			if (input[i,j]<cutoff_frac):
				output_array[i,j] = 0.0
			else:
				output_array[i,j]=1.0

	return output_array

def nPointOutput(size_coords_phase): #gaussian size parameters and some coordinates. 
	#a,x1,y1,x2,y2,p1,p2
	#size_coords_phase is a 4xN array structured as a, x, y, phase.

	Field = np.zeros([ImgResY, ImgResX])
	numdots = size_coords_phase.shape[0]

	for n in range(0,numdots):
		a = size_coords_phase[n,0]
		x = size_coords_phase[n,1]
		y = size_coords_phase[n,2] #casting these as their real parts because of the dtype of the array
		phase = size_coords_phase[n,3]
		print(phase)
		mask1 = np.exp(-a*((X-x)**2+(Y-y)**2)-1j*a*((X-x)**2+(Y-y)**2)) #add a gaussian amp profile
		mask1 = np.exp(phase*1j)*mask1# np.multiply(np.exp(phase*1j),mask1) #apply the phase factor
		Field = Field+mask1

	return Field #returns a complex field

def alt_iterative_Image_Plane(S_C_P, cutoff):
	#implementation of the iterative method by Wu Cheng and Tao.
	
	#DEFINE INPUT FIELD
	gaussian = True

	if (gaussian == True):
		input_U = np.exp(-.000001*((X)**2+(Y)**2)-1j*np.pi*.000001*((X)**2+(Y)**2))
		a = np.absolute(input_U)
		p = np.angle(input_U)
	else:
		a = np.ones([ImgResY, ImgResX])*1.0/2.0
		p = np.pi*np.ones([ImgResY, ImgResX]) #real valued phase angle


	im7 =axarr[0,0].matshow(a, cmap=plt.cm.Reds)
	axarr[0,0].set_title("Input Amp")
	fig.colorbar(im7, ax=axarr[0,0])
	im8 = axarr[0,1].matshow(p, cmap=plt.cm.Blues)
	axarr[0,1].set_title("Input Phase")
	fig.colorbar(im8, ax=axarr[0,1])

	
	#Gaussian Input
	#Just forget plotting input phase.... too problematic?
	#im6 = axarr[2,0].matshow(p)
	#axarr[2,0].set_title("Input Phase")
	#fig.colorbar(im6, ax=axarr[2,0])

	#DEFINE IDENTITY
	Ident = np.ones([ImgResY, ImgResX]) 
	#DEFINE CONSTRAINT MATRIX
	S = np.zeros([ImgResY, ImgResX])
	#try a half plane S
	for i in range(0,ImgResY):
	 	for j in range(0,ImgResX):
	 		if(j<ImgResX/2.0): #np.sqrt((j-ImgResY/2.0)**2+(i-ImgResX/2.0)**2)<100
	 			S[i,j] = 1

	S_flip = (Ident - S) #inversion of the constraint matrix

	#GENERATE THE TARGET FIELD
	target_field = nPointOutput(S_C_P) #COMPLEX Field
	A_target = np.absolute(target_field) #REAL AMPLITUDE
	P_target =  np.angle(target_field) #REAL ANGLE


	#NO PLOTTING FOR NOW
	im =axarr[1,0].matshow(A_target, cmap=plt.cm.Reds)
	axarr[1,0].set_title("Target Amp")
	fig.colorbar(im, ax=axarr[1,0])
	im2 = axarr[1,1].matshow(P_target, cmap=plt.cm.Blues)
	axarr[1,1].set_title("Target Phase")
	fig.colorbar(im2, ax=axarr[1,1])

	for n in range(0,cutoff): #the iterative mega-loop

		#calculate the complex amplitude in the IMAGE PLANE, ifft of input amplitude and current phase mask
		U_calculated = np.fft.ifft2(np.multiply(a,np.exp(1j*p))) #COMPLEX FIELD
		U_calculated = np.fft.ifftshift(U_calculated) #shifts k=0 to the center

		A_calculated = np.absolute(U_calculated) #REAL VALUED AMPLITUDE - IMAGE PLANE
		P_calculated = np.angle(U_calculated) #REAL VALUED PHASE ANGLE - IMAGE PLANE

		#Split the image plane into complementary planes

		#Calculate ALPHA plane values
		#Glues together part of the target and part of what we just calculated for both phase and amp
		A_alpha = np.multiply(A_target,S) + np.multiply(A_calculated,S_flip)#REAL VALUED AMPLITUDE
		P_alpha = np.multiply(P_target,S) + np.multiply(P_calculated,S_flip) #REAL VALUED PHASE ANGLE
		U_alpha = np.multiply(A_alpha, np.exp(1j*P_alpha)) #COMPLEX FIELD

		#Calculate BETA plane values - same procedure but invert the regions / how it is mixed
		A_beta = np.multiply(A_target,S_flip) + np.multiply(A_calculated,S) #REAL VALUED AMPLITUDE
		P_beta = np.multiply(P_target,S_flip) + np.multiply(P_calculated,S) #REAL VALUED PHASE
		U_beta = np.multiply(A_beta, np.exp(1j*P_beta)) #COMPLEX FIELD

		#the output fields are defined at this point,

		#propagate our new complex fields backwards into the FOURIER PLANE
		U_alpha_back = np.fft.fft2(U_alpha) #COMPLEX FIELD
		U_alpha_back = np.fft.fftshift(U_alpha_back) #Shift k=0 again
		P_alpha_back = np.angle(U_alpha_back) #REAL VALUED PHASE ANGLE

		U_beta_back = np.fft.fft2(U_beta) #COMPLEX FIELD
		U_beta_back = np.fft.fftshift(U_beta_back) 
		P_beta_back = np.angle(U_beta_back) #REAL VALUED PHASE ANGLE

		#generate the new phase array and iterate again - adds phase contributions from both
		p = np.angle((np.exp(1j*P_alpha_back)+np.exp(1j*P_beta_back))) #REAL VALUED PHASE ANGLE

	Image_Plane = U_alpha + U_beta #COMPLEX FIELD
	#toying with this
	output_amp = np.absolute(Image_Plane) #REAL VALUED AMPLITUDE
	output_phase = np.angle(Image_Plane) #REAL VALUED PHASE ANGLE

	#NO PLOTTING FOR NOW
	im3 = axarr[2,0].matshow(output_amp, cmap=plt.cm.Reds)
	axarr[2,0].set_title("Output Amp")
	fig.colorbar(im3, ax=axarr[2,0])

	im4 = axarr[2,1].matshow(output_phase, cmap=plt.cm.Blues)
	axarr[2,1].set_title("Output Phase")
	fig.colorbar(im4, ax=axarr[2,1],boundaries=phasebound)


	#Now for a sanity check, we take the input amplitude and p and propagate it
	fourier_plane = np.multiply(a,np.exp(1j*p))

	check_image_plane = np.fft.ifft2(fourier_plane) #COMPLEX FIELD
	check_image_plane = np.fft.ifftshift(check_image_plane) #shifts k=0 to the center

	check_image_plane_A = np.absolute(check_image_plane) #REAL VALUED AMPLITUDE - IMAGE PLANE
	check_image_plane_P = np.angle(check_image_plane) #REAL VALUED PHASE ANGLE - IMAGE PLANE

	output_check =axarr[3,0].matshow(check_image_plane_A, cmap=plt.cm.Greens,norm=LogNorm(vmin=0.0001, vmax=1))
	axarr[3,0].set_title("Sanity Check Amplitude")
	fig.colorbar(output_check, ax=axarr[3,0])

	output_check2 =axarr[3,1].matshow(check_image_plane_P, cmap=plt.cm.Greens)
	axarr[3,1].set_title("Sanity Check Phase")
	fig.colorbar(output_check2, ax=axarr[3,1])

	return p #return the final phase mask

if slmFlag == True:
    # create the object that handles the SLM array
    slm = slmpy.SLMdisplay(isImageLock = True)
    # retrieve SLM resolution (defined in monitor options)
    ImgResX, ImgResY = slm.getSize()
else:
    
    ImgResX = 512#792 #I think these need to be square
    ImgResY = 512#600

ImgCenterX = ImgResX/2
ImgCenterY = ImgResY/2

x = np.linspace(0,ImgResX,ImgResX)
y = np.linspace(0,ImgResY,ImgResY)

# initialize image matrix
X, Y = np.meshgrid(x,y)

X = X - ImgCenterX
Y = Y - ImgCenterY

fig, axarr = plt.subplots(4,2,sharex=True) #for plotting
phasebound = np.linspace(-np.pi,np.pi,100,endpoint=True)
phaseticks = np.linspace(-3.5,3.5,7,endpoint=True)

################################ USER INPUTS ########################################################
#DEFINE ARRAY OF POINTS, SIZE PARAM, X, Y, PHASE FACTOR
SCP = np.array([[.01,-100,0,np.pi/2.0],[.01,0,-50,0],[.01,100,0,np.pi]]) #,[.01,-100,0,np.pi/2],[.01,0,100,np.pi/2]
final_phase_mask = alt_iterative_Image_Plane(SCP,2000)
#####################################################################################################

#this is what we're going to hit now
if slmFlag != True:
	#im5 = axarr[2,1].matshow(final_phase_mask, cmap=plt.cm.Blues)
	#axarr[2,1].set_title("Phase Hologram")
	fig.subplots_adjust(right=.6)
	#fig.colorbar(im5, ax=axarr[2,1])

	new_ax = fig.add_axes([.55,0.3,0.4,0.4])
	new_ax.set_title("Phase Hologram")
	phaseholo = new_ax.matshow(final_phase_mask, cmap=plt.cm.Blues)
	fig.colorbar(phaseholo, cax=new_ax, boundaries=[-2*np.pi,2*np.pi])

	plt.show()
	#cv2.imshow('phase hologram',final_phase_mask)
	#cv2.waitKey()

