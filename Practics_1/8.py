# Виконайте перетворення значення температури у градусах Цельсія (C) для інших температурних
# шкал: Фаренгейта (F) і Кельвіна (K). Програма повинна відображати еквівалентну температуру у
#
# градусах Фаренгейта (F = 32 + 9/5 * C). Програма повинна відображати еквівалентну
# температуру у градусах Кельвіна (K = C + 273,15). Результати потрібно вивести на екран у
# відформатованому вигляді: з використанням двох знаків після десяткової крапки, мінімальною
# довжиною поля (15), вирівнюванням по центру. Зверніть увагу, у дійсних числах для розділення
# дробової і цілої частин використовують крапку.

# Користувач вводить значення температури у градусах Цельсія
temperature_celsius = float(input("Введіть температуру у градусах Цельсія: "))

# Конвертація температури у Фаренгейти та Кельвіни
temperature_fahrenheit = 32 + 9/5 * temperature_celsius
temperature_kelvin = temperature_celsius + 273.15

# Виведення відформатованих результатів
print(f"Фаренгейти: {temperature_fahrenheit:^15.2f}")
print(f"Кельвіни:   {temperature_kelvin:^15.2f}")