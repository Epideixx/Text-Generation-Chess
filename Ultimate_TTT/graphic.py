import pygame, sys
from game import TTT

pygame.init()

WIDTH = 600
HEIGHT = 600
LINE_COLOR = (23, 145, 135)
COLOR = (28, 170, 156)
RED = (255, 0, 0)

screen = pygame.display.set_mode(size = (WIDTH,HEIGHT ))
pygame.display.set_caption( 'Ultimate Tic-Tac-Toe')
screen.fill(color = COLOR)

def draw_lines():
    x = 0
    y = 0
    for i in range(1,9):
        x += HEIGHT//9
        if i%3 == 0:
            line_thickness = 10
        else : 
            line_thickness = 5

        pygame.draw.line(screen, LINE_COLOR, (x, 0), (x, 600), line_thickness)

    for j in range(1,9):
        y += WIDTH//9
        if j%3 == 0:
            line_thickness = 10
        else : 
            line_thickness = 5
        pygame.draw.line(screen, LINE_COLOR, (0, y), (600, y), line_thickness)

def draw_figures():
    for row in range(9):
        for col in range(9):
            pass

draw_lines()

player = 0

while True :
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

        if event.type == pygame.MOUSEBUTTONDOWN:

            mouseX = event.pos[0]
            mouseY = event.pos[1]

            clicked_row = int(mouseX // (HEIGHT//9))
            clicked_col = int(mouseY // (WIDTH//9))

            if player == 0:
                player = 1 - player
                pygame.draw.circle(screen, RED, center = (int(clicked_row * HEIGHT//9 + HEIGHT//18), int(clicked_col * WIDTH//9 + WIDTH//18)), radius = HEIGHT//25, width = 5)

            else:
                player = 1 - player
                pygame.draw.line(screen, RED, (int(clicked_row * HEIGHT//9 + HEIGHT//60), int(clicked_col * WIDTH//9 + WIDTH//60)), (int((1+clicked_row) * HEIGHT//9 - HEIGHT//60), int((clicked_col + 1) * WIDTH//9 - WIDTH//60)), 5)
                pygame.draw.line(screen, RED, (int((1+clicked_row) * HEIGHT//9 - HEIGHT//60), int(clicked_col * WIDTH//9 + WIDTH//60)), (int(clicked_row * HEIGHT//9 + HEIGHT//60), int((clicked_col + 1) * WIDTH//9 - WIDTH//60)), 5)
    
    pygame.display.update()