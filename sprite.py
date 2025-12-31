import pygame, util

focal_length = 200

class Sprite:
	def __init__(self, img, pos, height):
		self.pos = pos
		self.img = img
		self.height = height
		scale_factor = height / img.get_height()
		self.resized = pygame.transform.scale_by(img, scale_factor)
		self.dims = self.resized.get_size()

	def blit(self, screen, Z, offs):
		# coords relative to center of viewport
		x, y, z = self.pos
		rel = z - Z
		if rel <= 0: return
		scale = focal_length / rel

		dx, dy = (x*scale) - self.dims[0]/2, -(y*scale) - self.dims[1]/2
		if abs(dx) > util.W or abs(dy) > util.H: return
		nx, ny = util.W//2 + dx, util.H//2 + dy

		scaled = pygame.transform.scale_by(self.resized, scale)
		screen.blit(scaled, (nx+offs[0],ny+offs[1]))
