import cv2
from pylab import array, plot, show, axis, arange, figure, uint8

# Image data
image = cv2.imread('imgur.png',0) # load as 1-channel 8bit grayscale
cv2.imshow('image',image)
maxIntensity = 255.0 # depends on dtype of image data
x = arange(maxIntensity)

# Parameters for manipulating image data
phi = 1
theta = 1

# Increase intensity such that
# dark pixels become much brighter,
# bright pixels become slightly bright
newImage0 = (maxIntensity/phi)*(image/(maxIntensity/theta))**0.5
newImage0 = array(newImage0,dtype=uint8)

y = (maxIntensity/phi)*(x/(maxIntensity/theta))**0.5

# Decrease intensity such that
# dark pixels become much darker,
# bright pixels become slightly dark
newImage1 = (maxIntensity/phi)*(image/(maxIntensity/theta))**2
newImage1 = array(newImage1,dtype=uint8)

cv2.imshow('newImage1',newImage1)

z = (maxIntensity/phi)*(x/(maxIntensity/theta))**2