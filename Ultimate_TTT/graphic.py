import pygame, sys
from game import TTT

pygame.init()

WIDTH = 600
HEIGHT = 600
LINE_COLOR = (23, 145, 135)
COLOR = (28, 170, 156)
COLOR_POSSIBLE = tuple([c//2 for c in COLOR])
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


def play_game(player1 = "human", player2  ="human"):
    
    ttt = TTT()

    draw_lines()

    players = [player1, player2]
    player = 0

    while True :
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN and players[player] == "human":

                mouseX = event.pos[0]
                mouseY = event.pos[1]

                clicked_row = int(mouseX // (HEIGHT//9))
                clicked_col = int(mouseY // (WIDTH//9))

                move = (clicked_col, clicked_row)

                try : 
                    ttt.push(move)
                    
                    big_row = clicked_row // 3
                    big_col = clicked_col // 3
                    little_row = clicked_row % 3
                    little_col = clicked_col % 3

                    if ttt.upperBoard[big_col, big_row, 0] == 1:
                        if player == 0:
                            pygame.draw.circle(screen, RED, center = (int(big_row * HEIGHT//3 + HEIGHT//6), int(big_col * WIDTH//3 + WIDTH//6)), radius = HEIGHT//8, width = 5)

                        else:
                            pygame.draw.line(screen, RED, (int(big_row * HEIGHT//3 + HEIGHT//60), int(big_col * WIDTH//3 + WIDTH//60)), (int((1+big_row) * HEIGHT//3 - HEIGHT//60), int((big_col + 1) * WIDTH//3 - WIDTH//60)), 5)
                            pygame.draw.line(screen, RED, (int((1+big_row) * HEIGHT//3 - HEIGHT//60), int(big_col * WIDTH//3 + WIDTH//60)), (int(big_row * HEIGHT//3 + HEIGHT//60), int((big_col + 1) * WIDTH//3 - WIDTH//60)), 5)
                    
                    else :
                        s = pygame.Surface((HEIGHT//3, WIDTH//3 ))
                        s.set_alpha(128)
                        s.fill((255,255,255))
                        screen.blit(s, (little_row * HEIGHT//3, little_col * WIDTH//3))
                        # pygame.draw.rect(screen, color=RED, rect = (little_row * HEIGHT//3, little_col * WIDTH//3, HEIGHT//3, WIDTH//3 ))

                    if player == 0:
                        pygame.draw.circle(screen, RED, center = (int(clicked_row * HEIGHT//9 + HEIGHT//18), int(clicked_col * WIDTH//9 + WIDTH//18)), radius = HEIGHT//25, width = 5)

                    else:
                        pygame.draw.line(screen, RED, (int(clicked_row * HEIGHT//9 + HEIGHT//60), int(clicked_col * WIDTH//9 + WIDTH//60)), (int((1+clicked_row) * HEIGHT//9 - HEIGHT//60), int((clicked_col + 1) * WIDTH//9 - WIDTH//60)), 5)
                        pygame.draw.line(screen, RED, (int((1+clicked_row) * HEIGHT//9 - HEIGHT//60), int(clicked_col * WIDTH//9 + WIDTH//60)), (int(clicked_row * HEIGHT//9 + HEIGHT//60), int((clicked_col + 1) * WIDTH//9 - WIDTH//60)), 5)
                
                    player = 1 - player    
                    ttt = ttt.mirror()

    
                except Exception:
                    print("You cannot play this move")
            
            if players[player] == "random":
                clicked_col, clicked_row = ttt.playRandomMove()
                ttt = ttt.mirror()
                if player == 0:
                    pygame.draw.circle(screen, RED, center = (int(clicked_row * HEIGHT//9 + HEIGHT//18), int(clicked_col * WIDTH//9 + WIDTH//18)), radius = HEIGHT//25, width = 5)

                else:
                    pygame.draw.line(screen, RED, (int(clicked_row * HEIGHT//9 + HEIGHT//60), int(clicked_col * WIDTH//9 + WIDTH//60)), (int((1+clicked_row) * HEIGHT//9 - HEIGHT//60), int((clicked_col + 1) * WIDTH//9 - WIDTH//60)), 5)
                    pygame.draw.line(screen, RED, (int((1+clicked_row) * HEIGHT//9 - HEIGHT//60), int(clicked_col * WIDTH//9 + WIDTH//60)), (int(clicked_row * HEIGHT//9 + HEIGHT//60), int((clicked_col + 1) * WIDTH//9 - WIDTH//60)), 5)
            
                player = 1 - player

            if ttt.game_over == True:
                ttt.show()
                sys.exit()
        
        pygame.display.update()


if __name__ == '__main__':
    print(COLOR, tuple([c//2 for c in COLOR]))
    play_game(player2="human")
    