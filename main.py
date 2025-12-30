import pygame, numpy as np, sys
from scipy.signal import convolve2d
from PIL import Image
import math
import os

edge_threshold = 0.35 ** 2
bloom_threshold = 0.9
bloom_strength = 0.7

W, H = (1200, 904)
AW, AH = (int(W/8),int(H/8))

def load_ascii_map(filename):
	font_surface = pygame.image.load(filename)
	font_dims = font_surface.get_size()
	N = int(font_dims[0] / font_dims[1])
	pxsz = font_dims[1]

	buckets = []
	for x in range(N):
		sub = font_surface.subsurface((x*pxsz, 0, pxsz, font_dims[1]))
		buckets.append(pygame.surfarray.pixels3d(sub))
	return np.array(buckets)

ascii_levels = load_ascii_map("assets/fillASCII.png")
ascii_edges = load_ascii_map("assets/edgesASCII.png")

quad_to_ind = np.array([2, 3, 1, 4, 2, 3, 1, 4])

global G
global downscaled
G = np.empty((AW,AH), dtype=np.float32)
downscaled = np.empty((AW,AH))
levels = len(ascii_levels) - 1

# https://en.wikipedia.org/wiki/Relative_luminance
def vector_luma(pixels):
	srgb = pixels[..., :3] / 255.0
	lin_rgb = np.where(
		srgb < 0.04045,
		srgb / 12.92,
		((srgb + 0.055) / 1.055) ** 2.4
	)
	weights = np.array([0.2126, 0.7152, 0.0722])
	# out = np.sum(rgb * weights, axis=-1)
	return np.dot(lin_rgb, weights)

# https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics
def gaussian(sigma, x, y):
	s2 = sigma*sigma
	base = 1 / (2 * math.pi * s2)
	top = -1 * (x*x + y*y) / (2*s2)
	return base * math.exp(top)

# returns gaussian kernel with radius r
def gaussian_kernel(r):
	sigma = r/2
	s = (r*2)+1
	kernel = np.zeros((s,s))
	for x in range(-r, r+1, 1):
		for y in range(-r, r+1, 1):
			kernel[x+r, y+r]=gaussian(sigma, x,y)
	return kernel

gauss_5x5 = gaussian_kernel(2)

def convolve(img, kernel):
	channels = []
	for c in range(img.shape[2]):
		conv = convolve2d(img[..., c], kernel, mode='same')
		channels.append(conv)
	return np.stack(channels, axis=2)

def edge_detect(grayscale):
	blurred = convolve(grayscale[...,np.newaxis], gauss_5x5)
	# https://en.wikipedia.org/wiki/Sobel_operator
	sobel_kernel_x = np.array([	[-1, 0, 1],
															[-2, 0, 2],
															[-1, 0, 1]])
	sobel_kernel_y = np.array([	[-1, -2, -1],
															[0, 0, 0],
															[1, 2, 1]])

	gx = convolve(blurred, sobel_kernel_x).squeeze()
	gy = convolve(blurred, sobel_kernel_y).squeeze()
	return (gx, gy)

# element wise function
def grad_to_ascii(g):
	theta = math.atan2(gy[y, x], gx[y, x]) + math.pi
	quadrant = int(theta // (math.pi / 4))
	quad_to_ind = [2, 3, 1, 4, 2, 3, 1, 4]
	ind = quad_to_ind[quadrant]
	return ascii_edges[ind]


def min_max_normalize(arr):
	return (arr - arr.min()) / (arr.max() - arr.min())

def ascii_render(screen):
	global G
	global downscaled
	framebuffer = pygame.surfarray.pixels3d(screen)

	downscaled[:] = vector_luma(framebuffer)[::8, ::8]
	normalized = min_max_normalize(downscaled)
	B = (normalized * levels).astype(int)

	gx, gy = edge_detect(downscaled)
	G[:]=gx**2 + gy**2
	G /= G.max()

	theta = np.arctan2(gy, gx) + np.pi
	quadrant = (theta // (np.pi / 4)).astype(int) % 8
	ind = quad_to_ind[quadrant]

	edge_mask = (G > edge_threshold) & (B > 0)
	chars = ascii_levels[B]
	chars[edge_mask]=ascii_edges[ind][edge_mask]

	h, w = chars.shape[:2]
	framebuffer[:] = chars.transpose(0, 2, 1, 3, 4).reshape(h*8, w*8, 3)

def color_shader(surface):
	pixels = pygame.surfarray.pixels3d(surface)
	pixels[:, :, 1]=0
	pixels[:, :, 2]=0

def bloom_pass(surface):
	pixels = pygame.surfarray.pixels3d(surface)
	img = pixels.astype(np.float32) / 255.
	luma = vector_luma(pixels).squeeze() * 255
	knee = 0.1
	bright = np.clip((luma - bloom_threshold)/knee, 0, 1)

	gauss = gaussian_kernel(4)
	bloom = convolve(bright[..., np.newaxis], gauss)

	img += bloom * bloom_strength
	img = np.clip(img, 0, 1)
	pixels += (img * 255).astype(np.uint8)

def blit_character(screen, img, output_height, dest):
	resized = pygame.transform.scale_by(img, output_height / img.get_height())
	screen.blit(resized, dest)


assets = dict()
for asset in os.scandir("assets"):
	print("Loaded asset:", asset.name)
	k = asset.name.split('.')[0]
	assets[k]=pygame.image.load(asset.path)

pygame.init()
screen = pygame.display.set_mode((W,H))
clock = pygame.time.Clock()
running = True
t = 0
render_ascii = False
while running:
	for event in pygame.event.get():
			if event.type == pygame.QUIT: running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_a: render_ascii = not render_ascii

	print(clock.get_fps())
	t += clock.tick(30) / 1000

	screen.fill(0)
	blit_character(screen, assets["fool"], 1000, (0,0))
	blit_character(screen, assets["kent"], 1000, (600,0))

	if render_ascii: ascii_render(screen)
	# bloom_pass(screen)
	# color_shader(screen)

	pygame.display.flip()
