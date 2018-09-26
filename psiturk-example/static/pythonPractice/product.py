def product(numbers):
  mult = 1
  for number in numbers:
    index = numbers.index(number)
    mult *= numbers[index]
  return mult