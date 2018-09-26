''' This program will:
1. Roll a pair of dice.
2. Add the values of the roll.
3. Ask the user to guess a number.
4. Compare the user's guess to the total value.
5. Determine the winner (user or computer).'''

from random import randint
from time import sleep


def get_user_guess():
    guess = int(raw_input('Please enter your guess: '))
    return guess


def roll_dice(number_of_sides):
    first_roll = randint(1, number_of_sides)
    second_roll = randint(1, number_of_sides)
    max_val = number_of_sides * 2
    print 'The maximum possible value is %d' % (max_val)
    guess = get_user_guess()
    if guess > max_val:
        print 'Your guess is inconcievable'
    else:
        print 'Rolling...'
        sleep(1)
        print 'Rolling...'
        sleep(1)
        print 'Rolling...'
        sleep(2)
        print str(first_roll)
        sleep(1)
        print str(second_roll)
        sleep(1)
        total_roll = first_roll + second_roll
        print str(total_roll)
        print 'Result...'
        sleep(2)
        if guess == total_roll:
            print 'You win, congratulations! May the odds be ever in your favor.'
        else:
            print 'You lose. Better luck next time'


roll_dice(6)