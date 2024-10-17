# Виконайте переведення одиниць вимірювання відстаней. Значення відстані вказано у метрах. У
# кожному новому рядку програма виводить значення відстані, представлене у: дюймах, футах,
# милях, ярдах тощо. Числові дані на екрані мають бути у відформатованому вигляді: два знаки
# після десяткової крапки. Використайте функцію format(). Потрібні значення одиниць
# вимірювання знайдіть у мережі Інтернет.

METERS_TO_INCHES_RATIO = 39.3701
METERS_TO_FEET_RATIO = 3.28
METERS_TO_MILE_RATIO = 0.00062137

meters = input("Please, enter number of meters you would like to convert to Imperial measure system: ")

if meters.isnumeric():
    meters = int(meters)
    inches = round((meters * METERS_TO_INCHES_RATIO), 2)
    print("It is {finches} inch(es)".format(finches = inches))

    feet = round((meters * METERS_TO_FEET_RATIO), 2)
    print("It is {ffeet} foot(s)".format(ffeet=feet))

    mile = round((meters * METERS_TO_MILE_RATIO), 2)
    print("It is {fmile} mile(s)".format(fmile=mile))
else:
    print('Please enter a numeric value.')