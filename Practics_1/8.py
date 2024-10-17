# Користувач вводить значення температури у градусах Цельсія
temperature_celsius = float(input("Введіть температуру у градусах Цельсія: "))

# Конвертація температури у Фаренгейти та Кельвіни
temperature_fahrenheit = 32 + 9/5 * temperature_celsius
temperature_kelvin = temperature_celsius + 273.15

# Виведення відформатованих результатів
print(f"Фаренгейти: {temperature_fahrenheit:^15.2f}")
print(f"Кельвіни:   {temperature_kelvin:^15.2f}")