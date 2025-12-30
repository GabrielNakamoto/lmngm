import pygame, numpy as np, sys
from scipy.signal import convolve2d
from PIL import Image
import math

fool = "assets/fool.png"
kent = "assets/kent.png"
edge_threshold = 0.35

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
	rgb = pixels[..., :3]
	rgb = rgb ** 2.2
	weights = np.array([0.2126, 0.7152, 0.0722])
	# out = np.sum(rgb * weights, axis=-1)
	out = np.dot(rgb, weights)
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
	for c in range(img.shape[2]):
		conv = convolve2d(img[..., c], kernel, mode='same')
		channels.append(conv)
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

def ascii_render(screen, image_path, output_height, coords):
	print("Rendering:", image_path)
	img = Image.open(image_path)
	framebuffer = np.array(img)

	H, W, _ = framebuffer.shape
	N = int(H / output_height)

	gauss = gaussian_kernel(2)
	blurred = convolve(framebuffer, gauss)
	downscaled = blurred[::N, ::N]
	grayscale = vector_luma(downscaled)
	normalized = grayscale / grayscale.max()
	gx, gy = edge_detect(normalized)
	G = np.sqrt(gx*gx + gy*gy)
	G /= G.max()

	for y, row in enumerate(normalized.squeeze()):
		for x, L in enumerate(row):
			char = None
			B = round(L * (len(ascii_levels) - 1))
			if B == 0: continue
			if G[y, x] > edge_threshold:
				# theta is 0->2 pi, going counterclockwise from -x axis
				theta = math.atan2(gy[y, x], gx[y, x]) + math.pi
				quadrant = int(theta // (math.pi / 4))
				quad_to_ind = [2, 3, 1, 4, 2, 3, 1, 4]
				ind = quad_to_ind[quadrant]
				char = ascii_edges[ind]
			else:
				char = ascii_levels[B]
			try:
				screen.blit(char, (coords[0]+ x*8, coords[1] + y*8))
			except:
				pass

def color_shader(pixels):
	pixels[:, :, 1]=0
	pixels[:, :, 2]=0

bloom_threshold = 0.9
bloom_strength = 0.7
def bloom_pass(pixels):
	img = pixels.astype(np.float32) / 255.
	luma = vector_luma(pixels).squeeze()
	knee = 0.1
	bright = np.clip((luma - bloom_threshold)/knee, 0, 1)

	gauss = gaussian_kernel(4)
	bloom = convolve(bright[..., np.newaxis], gauss)

	img += bloom * bloom_strength
	img = np.clip(img, 0, 1)
	pixels += (img * 255).astype(np.uint8)

pygame.init()
screen = pygame.display.set_mode((1200, 900))
ascii_render(screen, fool, 1000 / 8, (0, 0))
ascii_render(screen, kent, 1000 / 8, (600, 0))

pixels = pygame.surfarray.pixels3d(screen)
color_shader(pixels)
bloom_pass(pixels)

pygame.display.flip()
clock = pygame.time.Clock()
running = True
while running:
	for event in pygame.event.get():
			if event.type == pygame.QUIT: running = False
	# pygame.display.flip()
	clock.tick(60)
