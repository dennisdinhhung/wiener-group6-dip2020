from cv2 import cv2 
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d


img = cv2.imread('dark.png')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def blur(img, kernel_size = 3):
	dummy = np.copy(img)
	h = np.eye(kernel_size) / kernel_size
	dummy = convolve2d(dummy, h, mode = 'valid')
	return dummy

def add_gaussian_noise(img, sigma):
	gauss = np.random.normal(0, sigma, np.shape(img))
	noisy_img = img + gauss
	noisy_img[noisy_img < 0] = 0
	noisy_img[noisy_img > 255] = 255
	return noisy_img

def gaussian_kernel(kernel_size = 3):
	h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
	h = np.dot(h, h.transpose())
	h /= np.sum(h)
	return h

def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy

gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

blur = blur(gray, kernel_size = 5)

noise = add_gaussian_noise(blur, sigma = 20)


kernel = gaussian_kernel(3)
wiener_img = wiener_filter(noise, kernel, K = 10)

kernel2 = gaussian_kernel(7)
wiener_img2 = wiener_filter(noise, kernel, K = 10)

plt.subplot(221), plt.imshow(gray, cmap="gray")
plt.subplot(222), plt.imshow(noise, cmap="gray")
plt.subplot(223), plt.imshow(wiener_img, cmap="gray")
plt.subplot(224), plt.imshow(wiener_img2, cmap="gray")
plt.show()