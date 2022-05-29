import pygame, sys
from game import TTT
from time import sleep
import os

from player_Transformer import Transfo_player

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

def draw_circles(circles):
    for (row, col) in circles:
        pygame.draw.circle(screen, RED, center = (int(row * HEIGHT//9 + HEIGHT//18), int(col * WIDTH//9 + WIDTH//18)), radius = HEIGHT//25, width = 5)

def draw_squares(squares):
    for (row, col) in squares:
        pygame.draw.line(screen, RED, (int(row * HEIGHT//9 + HEIGHT//60), int(col * WIDTH//9 + WIDTH//60)), (int((1+row) * HEIGHT//9 - HEIGHT//60), int((col + 1) * WIDTH//9 - WIDTH//60)), 5)
        pygame.draw.line(screen, RED, (int((1+row) * HEIGHT//9 - HEIGHT//60), int(col * WIDTH//9 + WIDTH//60)), (int(row * HEIGHT//9 + HEIGHT//60), int((col + 1) * WIDTH//9 - WIDTH//60)), 5) 

def draw_big_circles(circles):
    for (row, col) in circles:
        pygame.draw.circle(screen, RED, center = (int(row * HEIGHT//3 + HEIGHT//6), int(col * WIDTH//3 + WIDTH//6)), radius = HEIGHT//8, width = 5)

def draw_big_squares(squares):
    for (row, col) in squares:
        pygame.draw.line(screen, RED, (int(row * HEIGHT//3 + HEIGHT//60), int(col * WIDTH//3 + WIDTH//60)), (int((1+row) * HEIGHT//3 - HEIGHT//60), int((col + 1) * WIDTH//3 - WIDTH//60)), 8)
        pygame.draw.line(screen, RED, (int((1+row) * HEIGHT//3 - HEIGHT//60), int(col * WIDTH//3 + WIDTH//60)), (int(row * HEIGHT//3 + HEIGHT//60), int((col + 1) * WIDTH//3 - WIDTH//60)), 8)
                    

def play_game(player1 = "human", player2  ="human"):
    
    ttt = TTT()

    players = [player1, player2]
    player = 0
    squares = []
    circles = []
    big_squares = []
    big_circles = []
    area_to_play = None
    change_player = False
    wait = False

    name_folder_transfo = "Cluster_26_05"
    folder_path = os.path.join(os.path.dirname(__file__), name_folder_transfo)

    if "transformer" in players:
        transfos = [Transfo_player(folder_path), Transfo_player(folder_path)]
    
    mem_moves = []
    
    while True :
        
        if wait:
            sleep(0.5)
            wait = False

        screen.fill(color = COLOR)

        if area_to_play:
            s = pygame.Surface((HEIGHT//3, WIDTH//3 ))
            s.fill((0,0,0))
            s.set_alpha(128)
            screen.blit(s, (area_to_play[0] * HEIGHT//3, area_to_play[1] * WIDTH//3))
        
        else:
            s = pygame.Surface((HEIGHT, WIDTH))
            s.fill((0,0,0))
            s.set_alpha(128)
            screen.blit(s, (0,0))     

        draw_lines()
        draw_big_circles(big_circles)
        draw_big_squares(big_squares)
        draw_circles(circles)
        draw_squares(squares)

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
                            big_circles.append((big_row, big_col))
            
                        else:
                            big_squares.append((big_row, big_col))
                    
                    if ttt.upperBoard[little_col, little_row, 0] == 1 or ttt.upperBoard[little_col, little_row, 1]:
                        area_to_play = None
                    else :
                        area_to_play = (little_row, little_col)

                    if player == 0:
                        circles.append((clicked_row, clicked_col))

                    else:
                        squares.append((clicked_row, clicked_col))

                    change_player = True
                    mem_moves.append(ttt.rep_move(move))

                except Exception:
                    print("You cannot play this move")


            if players[player] in ["random", "transformer"]:

                if players[player] == "random":
                    clicked_col, clicked_row = ttt.playRandomMove()
                    mem_moves.append(ttt.rep_move((clicked_col, clicked_row)))

                elif players[player] == "transformer":
                    count_try = 0

                    board = ttt.rep_board()
                    previous_moves = " ".join(mem_moves)

                    while count_try <= 10:
                        moves = transfos[player].choose_move(board=board, previous_moves=previous_moves)[0]
                        print(moves)
                        print(previous_moves)
                        move = moves[0].split(" ")[-1].strip()
                        letters = {let: i for i, let in enumerate("ABCDEFGHI")}
                        move = (letters[move[0]], int(move[1]))
                        
                        try:
                            ttt.push(move)
                            mem_moves.append(ttt.rep_move(move))
                            clicked_col, clicked_row = move
                            break

                        except Exception:
                            print("Le coup n'est pas valable")
                        count_try += 1
                    
                    if count_try > 10:
                        clicked_col, clicked_row = ttt.playRandomMove()
                        mem_moves.append(ttt.rep_move((clicked_col, clicked_row)))


                if player == 0:
                    circles.append((clicked_row, clicked_col))

                else:
                    squares.append((clicked_row, clicked_col))
                
                big_row = clicked_row // 3
                big_col = clicked_col // 3
                little_row = clicked_row % 3
                little_col = clicked_col % 3

                if ttt.upperBoard[big_col, big_row, 0] == 1:
                    if player == 0:
                        big_circles.append((big_row, big_col))
        
                    else:
                        big_squares.append((big_row, big_col))
                
                if ttt.upperBoard[little_col, little_row, 0] == 1 or ttt.upperBoard[little_col, little_row, 1]:
                    area_to_play = None
                else :
                    area_to_play = (little_row, little_col)
                
                change_player = True
                wait = True
            
            if change_player:
                player = 1 - player    
                ttt = ttt.mirror()
                change_player = False
        
        pygame.display.update()

        if ttt.game_over == True:
            sleep(5)
            ttt.show()
            sys.exit()


if __name__ == '__main__':
    print(COLOR, tuple([c//2 for c in COLOR]))
    play_game(player2="transformer")
    