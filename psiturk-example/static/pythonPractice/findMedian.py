# this finds the median of a list
def median(input_list):
  input_list = sorted(input_list)
  input_length = len(input_list)
  if input_length % 2 != 0:
    index = input_length//2
    return input_list[index]
  else:
    index1 = input_length/2 - 1
    index2 = input_length/2
    mean = (input_list[index1] + input_list[index2])/2.0
    return mean

print median ([1,2,2,5,3,7])



