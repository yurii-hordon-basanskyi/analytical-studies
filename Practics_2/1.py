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
