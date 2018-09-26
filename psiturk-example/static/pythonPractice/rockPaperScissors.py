'''The program should do the following:

Prompt the user to select either Rock, Paper, or Scissors.
Instruct the computer to randomly select either Rock, Paper, or Scissors.
Compare the user's choice and the computer's choice.
Determine a winner (the user or the computer).
Inform the user who the winner is.'''

from random import randint

options = ['ROCK', 'PAPER', 'SCISSORS']
message = {'tie': 'Yawn, it is a tie!', 'won': 'Yay, you won!', 'lost': 'Aww you lost!'}


def decide_winner(user_choice, computer_choice):
    print 'You chose %s' % (user_choice)
    print 'Your opponent chose %s' % (computer_choice)
    if user_choice == computer_choice:
        print message['tie']
    elif user_choice == options[0]:
        if computer_choice == options[2]:
            print message['lost']
        elif computer_choice == options[1]:
            print message['won']
    elif user_choice == options[1]:
        if computer_choice == options[2]:
            print message['lost']
        elif computer_choice == options[0]:
            print message['won']
    elif user_choice == options[2]:
        if computer_choice == options[0]:
            print message['lost']
        elif computer_choice == options[1]:
            print message['won']


def play_RPS():
    user_choice = raw_input('Enter ROCK, PAPER, or SCISSORS: ')
    user_choice = user_choice.upper()
    computer_choice = options[randint(0, 2)]
    decide_winner(user_choice, computer_choice)


play_RPS()