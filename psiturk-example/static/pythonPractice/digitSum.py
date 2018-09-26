# this program sums all the numbers that are put into it

def digit_sum(n):
  n = str(n)
  addition = 0
  for i in range(len(n)):
    addition += int(n[i])
  return addition
#n = 1234
#n = '1234'
#addition = 0
#n[0] = '1', n[1] = '2', n[2] = '3', n[3] = '4'
#int(all the above)
#n[0] = 1, n[1] = 2, n[2] = 3, n[3] = 4
#for x (index #) in range(len(n)) -> for x in range(4) -> for x between 0 and 3
# addition = 0 + n[0]
# addition = additon + n[1] + n[2] + n[3]
#

print digit_sum(1234)