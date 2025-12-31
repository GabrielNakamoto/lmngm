from util import load_ascii_map
from scipy.signal import convolve2d
import pygame, numpy as np, sys
import math, util

global G
global downscaled
G = np.empty((util.AW,util.AH), dtype=np.float32)
downscaled = np.empty((util.AW,util.AH))

ascii_levels = load_ascii_map("assets/fillASCII.png")
ascii_edges = load_ascii_map("assets/edgesASCII.png")
edge_threshold = 0.35 ** 2
quad_to_ind = np.array([2, 3, 1, 4, 2, 3, 1, 4])
levels = len(ascii_levels) - 1

# https://en.wikipedia.org/wiki/Relative_luminance
luma_weights = np.array([0.2126, 0.7152, 0.0722])
def vector_luma(pixels):
	srgb = pixels[..., :3] / 255.0
	lin = np.where(srgb < 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
	return np.dot(lin, luma_weights)

# https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics
def gaussian_kernel(r):
	s2, s = (r/2)**2, (r*2)+1
	x, y = np.arange(-r, r+1), np.arange(-r, r+1)[:, None]
	return np.exp(-(x**2 + y**2) / (2*s2)) / (2 * math.pi * s2)
gauss_5x5 = gaussian_kernel(2)

def convolve(img, kernel):
	return np.stack([convolve2d(img[...,c], kernel, mode='same')
		for c in range(img.shape[2])], axis=2)

# https://en.wikipedia.org/wiki/Sobel_operator
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
def edge_detect(grayscale):
	blurred = convolve(grayscale[...,np.newaxis], gauss_5x5)
	return (convolve(blurred, sobel_kernel_x).squeeze(), convolve(blurred, sobel_kernel_y).squeeze())

def min_max_normalize(arr): return (arr - arr.min()) / (arr.max() - arr.min())

def ascii_render(screen):
	global G
	global downscaled
	framebuffer = pygame.surfarray.pixels3d(screen)

	downscaled[:] = vector_luma(framebuffer)[::8, ::8]
	B = (min_max_normalize(downscaled) * levels).astype(int)
	gx, gy = edge_detect(downscaled)
	G[:]=gx**2 + gy**2; G /= G.max()

	theta = np.arctan2(gy, gx) + np.pi
	quadrant = (theta // (np.pi / 4)).astype(int) % 8

	edge_mask = (G > edge_threshold) & (B > 0)
	chars = ascii_levels[B]
	chars[edge_mask]=ascii_edges[quad_to_ind[quadrant]][edge_mask]

	h, w = chars.shape[:2]
	framebuffer[:] = chars.transpose(0, 2, 1, 3, 4).reshape(h*8, w*8, 3)

def color_shader(surface, weights):
    pixels = pygame.surfarray.pixels3d(surface)
    rw, gw, bw = weights
    pixels[..., 0] = np.minimum(pixels[..., 0] * rw, 255)
    pixels[..., 1] = np.minimum(pixels[..., 1] * gw, 255)
    pixels[..., 2] = np.minimum(pixels[..., 2] * bw, 255)

bloom_threshold = 0.9
bloom_strength = 0.7
def bloom_pass(surface):
	pixels = pygame.surfarray.pixels3d(surface)
	img = pixels.astype(np.float32) / 255.
	luma = vector_luma(pixels).squeeze() * 255
	bright = np.clip((luma - bloom_threshold)/0.1, 0, 1)
	bloom = convolve(bright[..., np.newaxis], gauss_5x5)
	img = np.clip(img + bloom * bloom_strength, 0, 1)
	pixels += (img * 255).astype(np.uint8)

def brighten(surface, factor=1.5):
    pixels = pygame.surfarray.pixels3d(surface)
    pixels[:] = np.clip(pixels * factor, 0, 255).astype(np.uint8)
