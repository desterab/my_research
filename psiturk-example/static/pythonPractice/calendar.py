'''The program should behave in the following way:

1. Print a welcome message to the user
2. Prompt the user to view, add, update, or delete an event on the calendar
3. Depending on the user's input: view, add, update, or delete an event on the calendar
4. The program should never terminate unless the user decides to exit'''

from time import sleep, strftime

user_name = 'Abby D.'
calendar = {}


def welcome():
    print 'Hello, %s, and welcome.' % (user_name)
    print 'Your calendar is opening...'
    sleep(3)
    print 'Today is ' + strftime('%A %B %d, %Y') + '.'
    print 'The time is ' + strftime('%H:%M:%S') + '.'
    sleep(3)
    print 'What would you like to do?'


def start_cal():
    welcome()
    start = True
    while start:
        user_choice = raw_input('A to Add, U to Update, V to View, D to Delete, X to Exit: ')
        user_choice = user_choice.upper()
        if user_choice == 'V':
            if len(calendar.keys()) < 1:
                print 'Calendar empty.'
            else:
                print calendar

        elif user_choice == 'U':
            date = raw_input('What is the date?  ')
            update = raw_input('Enter the update:  ')
            calendar[date] = update
            print 'Update successful.'
            print calendar

        elif user_choice == 'A':
            event = raw_input('Enter an event:  ')
            date = raw_input('Enter date (MM/DD/YYYY):  ')
            if (len(date) > 10 or int(date[6:]) < int(strftime('%Y'))):
                print 'Invalid Date'
                try_again = raw_input('Try again? Y for Yes, N for No:  ')
                try_again = try_again.upper()
                if try_again == 'Y':
                    continue
                else:
                    start = False
            else:
                calendar[date] = event
                print 'Success!'
                sleep(1)
                print calendar

        elif user_choice == 'D':
            if len(calendar.keys()) < 1:
                print 'Calendar is empty'
            else:
                event = raw_input('Which event?  ')
                for date in calendar.keys():
                    if event == calendar[date]:
                        del calendar[date]
                        print 'Successfully deleted!'
                        print calendar
                    else:
                        print 'That event does not exist.'

        elif user_choice == 'X':
            start = False
        else:
            print 'Invalid command. How rude.'
            start = False


start_cal()

