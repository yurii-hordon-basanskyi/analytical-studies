user_input = input("Введіть чотирицифрове число: ")

if len(user_input) == 4 and user_input.isnumeric():
    digits = [int(digit) for digit in user_input]
    digits_sum = sum(digits)
    print(f"{' + '.join(user_input)} = {digits_sum}")
else:
    print("Ввід невірний")