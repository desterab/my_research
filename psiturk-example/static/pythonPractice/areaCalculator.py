# This program should prompt the user to select a shape, calculate the area of that shape, and print the area of that shape to the user.
print 'Starting up...'

name = raw_input("What's your name? ")
option = raw_input('Enter C for circle or T for triangle: ')

if option == 'C':
    radius = float(raw_input('Input radius: '))
    area = 3.14159 * (radius ** 2)
    print str(area)
    elif option == 'T':
    base = float(raw_input('Please input the base: '))
height = float(raw_input('Please input the height: '))
area = (base * height) / 2
print str(area)
else:
print 'You have entered an invalid shape'

print 'Shutting down...'