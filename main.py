import pygame, numpy as np, sys
from scipy.signal import convolve2d
from PIL import Image
import math

fool = np.array(Image.open("assets/fool.png"))
kent = np.array(Image.open("assets/kent.png"))
castle = "assets/castle.jpg"
tempest = "assets/tempest.jpg"

edge_threshold = 0.35
bloom_threshold = 0.9
bloom_strength = 0.7

# Eye close effect

def load_ascii_map(filename):
	font_surface = pygame.image.load(filename)
	font_dims = font_surface.get_size()
	N = int(font_dims[0] / font_dims[1])
	pxsz = font_dims[1]

	buckets = []
	for x in range(N):
		sub = font_surface.subsurface((x*pxsz, 0, pxsz, font_dims[1]))
		buckets.append(sub)
	return buckets

ascii_levels = load_ascii_map("assets/fillASCII.png")
ascii_edges = load_ascii_map("assets/edgesASCII.png")

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
	out = np.dot(lin_rgb, weights)
	return out[..., np.newaxis]

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

def convolve(img, kernel):
	channels = []
	for c in range(min(3,img.shape[2])):
		conv = convolve2d(img[..., c], kernel, mode='same')
		channels.append(conv)
	if img.shape[2] > 3: channels.append(img[:, :, 3])
	return np.stack(channels, axis=2)

def edge_detect(grayscale):
	# https://en.wikipedia.org/wiki/Sobel_operator
	sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
	sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

	gauss = gaussian_kernel(2)

	blurred = convolve(grayscale, gauss)
	gx = convolve(blurred, sobel_kernel_x).squeeze()
	gy = convolve(blurred, sobel_kernel_y).squeeze()

	return (gx, gy)

def ascii_render(screen, framebuffer, output_height, coords, gamma=1, test=False):
	H, W, C = framebuffer.shape
	N = int(H / output_height)

	# Pre-processing
	gauss = gaussian_kernel(2)
	blurred = convolve(framebuffer, gauss)
	downscaled = blurred[::N, ::N]
	luma = vector_luma(downscaled)
	normalized = (luma - luma.min()) / (luma.max() - luma.min()) # clamp 0->1
	normalized **= gamma
	gx, gy = edge_detect(luma)
	G = np.sqrt(gx*gx + gy*gy)
	G /= G.max()
	
	# Render ascii to framebuffer
	for y, row in enumerate(normalized.squeeze()):
		for x, L in enumerate(row):
			char = None
			B = round(L * (len(ascii_levels) - 1))
			if test: B = 0
			if C > 3 and downscaled[y, x, 3] == 0: continue
			if G[y, x] > edge_threshold and B > 0:
				# theta is 0->2 pi, going counterclockwise from -x axis
				theta = math.atan2(gy[y, x], gx[y, x]) + math.pi
				quadrant = int(theta // (math.pi / 4))
				quad_to_ind = [2, 3, 1, 4, 2, 3, 1, 4]
				ind = quad_to_ind[quadrant]
				char = ascii_edges[ind]
			else:
				char = ascii_levels[B]
			try:
				# apply color shader to char surface first using downscale rgb value
				screen.blit(char, (coords[0]+ x*8, coords[1] + y*8))
			except:
				pass # Ignore off screen coords for now

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

"""
I need to rework so I just resize and blit characters to screen
and do single shader pass on entire frame buffer
"""

pygame.init()
screen = pygame.display.set_mode((1200, 900))
# pygame.image.save(screen, "render.png")
# pygame.display.flip()
clock = pygame.time.Clock()
running = True
while running:
	for event in pygame.event.get():
			if event.type == pygame.QUIT: running = False

	screen.fill(0)
	# ascii_render(screen, "assets/reach.png", 600 / 8, (0, 0))
	# ascii_render(screen, tempest, 1000 / 8, (0, 0), gamma=1/1.2)
	# ascii_render(screen, "assets/child-peasants.png", 1000/8, (0,0), test=False)
	# ascii_render(screen, "assets/peasants2.png", 1000/8, (500,-400), test=False)
	# ascii_render(screen, castle, 1000 / 8, (0, 0), gamma=1/1.4)
	ascii_render(screen, kent, 1000 / 8, (600, 0))
	ascii_render(screen, fool, 1000 / 8, (0, 0))
	color_shader(screen)
	bloom_pass(screen)

	pygame.display.flip()
	clock.tick(60)
