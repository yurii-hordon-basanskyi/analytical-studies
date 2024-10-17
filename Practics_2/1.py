# Збережіть назви мов світу (Ukrainian, French, Bulgarian, Norwegian, Latvian або інші)
# у списку. Простежте за тим, щоб елементи у списку не зберігались в алфавітному порядку.
# Застосуйте функції sorted(), reverse(), sort() до списку. Виведіть список на екран до і
# після використання кожної із функцій.

languages = ["Ukrainian", "French", "Bulgarian", "Norwegian", "Latvian"]

def print_original_list():
    print("\nОригінальний список:")
    print(languages)

print_original_list()
print("\nСписок після застосування функції sorted():")
print(sorted(languages))

print_original_list()
languages.reverse()
print("\nСписок після застосування функції reverse():")
print(languages)

print_original_list()
languages.sort()
print("\nСписок після застосування функції sort():")
print(languages)
