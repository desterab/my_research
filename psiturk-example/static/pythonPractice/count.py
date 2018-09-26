#this program counts how many of the item there are

def count(sequence, item):
    how_many = 0
    for thing in sequence:

        if thing == item:
            how_many += 1
    return how_many


print count([1, 2, 1, 1], 1)