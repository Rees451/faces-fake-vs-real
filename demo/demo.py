'''
Demo of the classification algorithm in a game format - can you beat the
algorithm at spotting the fake pictures?
'''

# Imports Section -------------------------------------------------------------
import pygame
import os
import sys
import numpy as np

# Get the full path of the 'fake vs real' project directory
# if 'fake-vs-real' or 'Faces' not in os.getcwd():
#     project_path = os.path.abspath(os.getcwd() + '/faces-fake-vs-real')
# else:
project_path = os.getcwd()

# Append project path to sys.path to access files/trained models, NOQA
# stops flake8 complaining
sys.path.append(project_path + '/src')
sys.path.append('../src')
sys.path.append('./src')
import trained_model as tm  # NOQA
import loading as ld  # NOQA

# File locations
model_dir = project_path + '/models/gs_results'
model_id = 'best_c1_gs'

# Game Section ----------------------------------------------------------------
pygame.init()

# width and hight of game window
width = 1000
height = 600
game_display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Photospot')

# Constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (220, 220, 220)
FONT = pygame.font.Font("freesansbold.ttf", 40)
GAME_IMG_SIZE = (250, 250)

# Randomly pick n images and return the paths using the loading file
fake_paths, real_paths = ld.get_paths(
    os.path.abspath(f'{project_path}/data/processed/sf/all/test'),
    'hard',
    n=10)

# Load images into pygame, keep the path in a tuple
fakes = [(path, pygame.image.load(path)) for path in fake_paths]
reals = [(path, pygame.image.load(path)) for path in real_paths]

# Transform the images to suitable size for screen
fakes = [(path, pygame.transform.scale(img, GAME_IMG_SIZE))
         for path, img in fakes]
reals = [(path, pygame.transform.scale(img, GAME_IMG_SIZE))
         for path, img in reals]


# Classes for the game
# ImgPair class tracks the image that are shown on screen
class ImgPair():
    def __init__(self, fakes, reals):
        x = (width * 0.25)
        y = (height / 2)
        self.left_loc = (x, y)
        self.right_loc = (width - GAME_IMG_SIZE[0], y)
        self.fakes = fakes
        self.reals = reals
        self.pair_generator = iter(self._pair_generator())
        # Initalise a pair of images to show
        self.get_next()

    def blit_imgs(self, game_display):
        left_rect = self.left[1].get_rect(center=self.left_loc)
        # left_rect.center(self.left_loc[0], self.left_loc[1])
        right_rect = self.right[1].get_rect(center=self.right_loc)
        # right_rect.center(self.right_loc[0], self.right_loc[1])
        game_display.blit(self.left[1], left_rect)
        game_display.blit(self.right[1], right_rect)

    def get_next(self):
        try:
            self.left, self.right, self.fake_on = next(self.pair_generator)
        except Exception:
            self.left, self.right, self.fake_on = None, None, None

    def _pair_generator(self):
        for fake, real in zip(self.fakes, self.reals):
            fake_on = np.random.choice(['left', 'right'])

            if fake_on == 'left':
                yield fake, real, fake_on
            else:
                yield real, fake, fake_on


# Player reprents the user, (the algo also inherits from player
# to gain its functionality)
class Player():
    def __init__(self):
        self.questions = 0
        self.score = 0

    def add_to_score(self, marking='correct'):
        self.questions += 1
        if marking == 'correct':
            self.score += 1

    def print_score(self):
        print(f'You scored{self.score} out of {self.questions} questions')

    def submit_answer(self, answer, img_pair):
        if answer == 'left' and img_pair.fake_on == 'left':
            self.add_to_score('correct')
            result = 'correct'
        elif answer == 'right' and img_pair.fake_on == 'right':
            self.add_to_score('correct')
            result = 'correct'
        else:
            self.add_to_score('incorrect')
            result = 'inccorect'

        print(f'Selected {answer}, which is {result}')

    def reset_scores(self):
        self.score = 0
        self.questions = 0


# Algo comines the Player class and the trained model to make
# predictions on which image is fake
class Algo(Player, tm.TrainedModel):
    def __init__(self):
        Player.__init__(self)
        tm.TrainedModel.__init__(self, model_dir, model_id)

    def get_prediction(self, img_pair):
        answer = self.spot_from_pair(img_pair.left[0], img_pair.right[0])
        self.submit_answer(answer, img_pair)


# Scoreboard gets the current scores from the players (algo and
# human and shows them on screen)
class Scoreboard():
    def __init__(self):
        pass

    def _blit_score(self, msg, pos):
        text_surf = FONT.render(msg, True, BLACK)

        # You can pass the center directly to the `get_rect` method.
        if pos == 'left':
            pos_x = width / 4
        elif pos == 'right':
            pos_x = width * 3 / 4

        text_rect = text_surf.get_rect(center=(pos_x, 30))
        game_display.blit(text_surf, text_rect)

    def blit_scores(self, player, algo):
        player_msg = f'Your score: {player.score} / {player.questions}'
        algo_msg = f'Algo score: {algo.score} / {algo.questions}'
        self._blit_score(player_msg, pos='left')
        self._blit_score(algo_msg, pos='right')


# Function to check which key has been pressed
def check_selection(event):
    selected = None

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            selected = 'left'
        if event.key == pygame.K_RIGHT:
            selected = 'right'

    return selected


def start_loop():
    pass


def end_game(scoreboard, player, algo):
    clock = pygame.time.Clock()
    done = False
    restart = False

    while not done:

        # Game display updates
        game_display.fill(GRAY)

        text_surf = FONT.render('Press Enter to play again, or esc to quit',
                                True, BLACK)
        text_rect = text_surf.get_rect(center=(width / 2, height / 2))
        game_display.blit(text_surf, text_rect)
        scoreboard.blit_scores(player, algo)
        # Update the display and tick the clock
        pygame.display.update()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    done = True
                    restart = True
                elif event.key == pygame.K_ESCAPE:
                    done = True
            elif event.type == pygame.QUIT:
                done = True

    return restart


def setup():
    # Initialise the players and scoreboard and img pairs
    player = Player()
    algo = Algo()
    scoreboard = Scoreboard()
    img_pair = ImgPair(fakes, reals)

    return player, algo, scoreboard, img_pair


# Main game loop --------------------------------------------------------------
def main_loop():
    clock = pygame.time.Clock()

    # Initialise the players and scoreboard and img pairs
    player, algo, scoreboard, img_pair = setup()

    done = False
    while not done:

        # Game display updates
        game_display.fill(GRAY)
        # Blit the imgs
        img_pair.blit_imgs(game_display)
        # Blit the scoreboard
        scoreboard.blit_scores(player, algo)

        # Blit instructions
        font = pygame.font.Font('freesansbold.ttf', 32)
        text = font.render('Select which image is fake using arrow keys', True,
                           BLACK)
        textRect = text.get_rect()
        textRect.center = (width // 2, height - 40)
        game_display.blit(text, textRect)

        # Update the display and tick the clock
        pygame.display.update()
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True

            # Get whether the left or right image was chosen
            selected = check_selection(event)
            if selected is not None:
                # Check the players answer
                player.submit_answer(selected, img_pair)
                # Check the algos answer
                algo.get_prediction(img_pair)
                # Get the next iamge pair
                img_pair.get_next()

        if img_pair.left is None:
            restart = end_game(scoreboard, player, algo)
            if restart:
                player, algo, scoreboard, img_pair = setup()
            else:
                done = True
            # Else if done is True game will exit


# Run the main_loop and quit it the loop ends
main_loop()
pygame.quit()
