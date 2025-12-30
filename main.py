import pygame, numpy as np, sys
from scipy.signal import convolve2d
from PIL import Image
import math

font_surface = pygame.image.load("assets/fillASCII.png")
font_dims = font_surface.get_size()
buckets = int(font_dims[0] / font_dims[1])
pxsz = font_dims[1]

ascii_levels = []
for x in range(buckets):
	sub = font_surface.subsurface((x*pxsz, 0, pxsz, font_dims[1]))
	ascii_levels.append(sub)

fool = "assets/fool.png"
kent = "assets/kent.png"

# https://en.wikipedia.org/wiki/Sobel_operator
sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def vector_luma(pixels):
	rgb = pixels[..., :3]
	rgb = rgb ** 2.2
	weights = np.array([0.2126, 0.7152, 0.0722])
	out = np.sum(rgb * weights, axis=-1)
	return out[..., np.newaxis]

# https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics
def gaussian(sigma, x, y):
	s2 = sigma*sigma
	base = 1 / (2 * math.pi * s2)
	top = -1 * (x*x + y*y) / (2*s2)
	return base * math.exp(top)

# returns gaussian kernel with radius r
def gaussian_kernel(sigma, r):
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

def edge_detect(img_path):
	kernel = gaussian_kernel(1, 2)

	img = np.array(Image.open(img_path))
	grayscale = vector_luma(img)

	blurred = convolve(grayscale, kernel)
	gx = convolve(blurred, sobel_kernel_x)
	gy = convolve(blurred, sobel_kernel_y)

	gradient = np.sqrt(gx*gx + gy*gy)
	gradient = gradient.squeeze() # remove channel dim
	gradient /= gradient.max() # normalize gradient

	result = np.where(gradient > 0.25, gradient, 0)
	result = (result * 255).astype(np.uint8)
	output = Image.fromarray(result)
	output.show()

# https://en.wikipedia.org/wiki/Relative_luminance
def luma(pixel):
	r, g, b, a = pixel
	r = r ** 2.2
	g = g ** 2.2
	b = b ** 2.2
	return 0.2126*r + 0.7152*g + 0.0722*b

def ascii_render(image_path, output_height, coords):
	print("Rendering:", image_path)
	img = Image.open(image_path)
	framebuffer = np.array(img)
	print("Raw shape:", framebuffer.shape)
	H, W, _ = framebuffer.shape

	# downscale factor
	N = int(H / output_height)
	print("Downscale:", N)

	nW = W//N
	nH = H//N
	print("New dims:", nH, nW)

	# Average NxN blocks to a single pixel
	newbuf = np.zeros((nH, nW))
	for i in range(nH):
		for j in range(nW):
			block = framebuffer[i*N:(i+1)*N, j*N:(j+1)*N, :]
			avg = 0
			for p in block.reshape(-1, 4):
				avg += luma(p)
			avg /= N*N
			newbuf[i, j]=avg
	brightest = newbuf.max()

	for y, row in enumerate(newbuf):
		for x, l in enumerate(row):
			normalized = l / brightest
			ind = round(normalized * (len(ascii_levels) - 1))
			screen.blit(ascii_levels[ind], (coords[0]+ x*8, coords[1] + y*8))
	print()


edge_detect(fool)
"""
pygame.init()
screen = pygame.display.set_mode((1200, 900))
ascii_render(fool, 900 / 8, (0, 0))
ascii_render(kent, 900 / 8, (600, 0))

pygame.display.flip()
clock = pygame.time.Clock()
running = True
while running:
	for event in pygame.event.get():
			if event.type == pygame.QUIT: running = False
	# pygame.display.flip()
	clock.tick(60)
"""
