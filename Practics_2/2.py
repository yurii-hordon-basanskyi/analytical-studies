input_string = input("Введіть числа розділені пропусками: ")

numbers = list(map(int, input_string.split()))
total_sum = sum(numbers)
print(total_sum)