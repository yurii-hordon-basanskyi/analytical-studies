# Виконайте розкладання чотирицифрового цілого числа і виведіть на екран суму цифр у числі.
# Наприклад, якщо обрали число 6259, то програма повинна вивести на екран повідомлення: 6 +
# 2 + 5 + 9 = 22. Використайте функцію format() для відображення результату або f-
# рядки.

user_input = input("Введіть чотирицифрове число: ")

if len(user_input) == 4 and user_input.isnumeric():
    digits = [int(digit) for digit in user_input]
    digits_sum = sum(digits)
    print(f"{' + '.join(user_input)} = {digits_sum}")
else:
    print("Ввід невірний")