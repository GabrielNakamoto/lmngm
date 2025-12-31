from util import load_assets, W, H, Subtitle
from render import brighten, ascii_render, color_shader, bloom_pass
from sprite import Sprite
import pygame, sys, math

assets = load_assets()

pygame.init()
screen = pygame.display.set_mode((W,H))
clock = pygame.time.Clock()
running = True
t = 0
z = 0

storm_level = 0.1
storm_factor = 25
storming = True

brighten(assets["tempest"])
brighten(assets["castle"])
brighten(assets["child-peasants"])
render_ascii = True

# peasant = Sprite(assets["child-peasants"], (20,100,200), 500)
peasant = Sprite(assets["child-peasants"], (30,100,300), 500)
fool = Sprite(assets["fool"], (-200,100,100), 500)
kent = Sprite(assets["kent"], (150,100,150), 500)
to_render = []
# to_render.append(peasant)
to_render.append(fool)
to_render.append(kent)
dt = 0
subs = []
subs.append(Subtitle("Lear: Crack cheeks or something", 3))
subs.append(Subtitle("Lear: Poor naked wretches", 3))
while running:
	# print(f"{clock.get_fps():.3f}", end='\r', flush=True)
	t += dt 
	for event in pygame.event.get():
			if event.type == pygame.QUIT: running = False
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_a: render_ascii = not render_ascii
		
	keys = pygame.key.get_pressed()
	if keys[pygame.K_w]:
		z += dt*40
		storm_level = min(1.0, storm_level + dt*0.1)

	screen.fill((0,0,0))
	offset = (
			math.sin(t*2*math.pi*10) * storm_level * storm_factor,
			math.cos(t*2*math.pi*10) * storm_level * storm_factor
		) if storming else (0,0)
	for s in to_render: s.blit(screen, z, offset)

	if render_ascii: ascii_render(screen)
	# bloom_pass(screen)
	# color_shader(screen, (1,0,0))
	color_shader(screen, (1, 1-storm_level, 1-storm_level))
	# blit_character(screen, assets["fool"], 1000, (0,0))

	if len(subs) > 0:
		front = subs[0]
		if front.start_time == 0: front.start_time = t
		front.render(screen)
		if t - front.start_time >= front.duration: subs=subs[1:]

	pygame.display.flip()
	dt = clock.tick(60) / 1e3
pygame.quit()
