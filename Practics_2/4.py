user_input = input ("Введіть пʼять чисел розділених пропуском: ")

numbers = list(user_input.split())
print(numbers)

sorted_numbers = numbers
sorted_numbers.reverse()

print(sorted_numbers)

print(''.join(sorted_numbers))