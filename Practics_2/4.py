# Необхідно зчитати рядок з 5 цифр, розділених пропусками, і зберегти кожну цифру у список.
# Створіть копію списку із впорядкованими елементами у зворотному порядку. Виведіть число,
# яке утворюється об’єднанням елементів нового списку.

user_input = input ("Введіть пʼять чисел розділених пропуском: ")

numbers = list(user_input.split())
print(numbers)

sorted_numbers = numbers
sorted_numbers.reverse()

print(sorted_numbers)

print(''.join(sorted_numbers))