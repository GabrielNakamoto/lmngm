import numpy as np, os, pygame

W, H = (1200, 904)
AW, AH = (int(W/8),int(H/8))

def load_ascii_map(filename):
	font_surface = pygame.image.load(filename)
	font_dims = font_surface.get_size()
	buckets = [font_surface.subsurface((x*font_dims[1], 0, font_dims[1], font_dims[1]))
		for x in range(int(font_dims[0]/font_dims[1]))]
	return np.array(list(map(pygame.surfarray.pixels3d, buckets)))

def load_assets(asset_dir="assets"):
	assets = dict()
	for asset in os.scandir(asset_dir):
		if "ASCII" in asset.name: continue
		print("Loaded asset:", asset.name)
		k = asset.name.split('.')[0]
		assets[k]=pygame.image.load(asset.path)
	return assets

class Subtitle:
	def __init__(self, text, duration):
		self.text = text
		self.duration = duration
		self.start_time = 0
	def render(self, screen):
		font = pygame.font.SysFont("arial", 32)
		surf = font.render(self.text, True, (255,255,255))
		rect = surf.get_rect(center=(W//2, H - 60))

		screen.blit(surf, rect)
