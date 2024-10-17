number_of_days = int(input("Введіть кількість днів канікул: "))

total_hours = number_of_days * 24
total_minutes = total_hours * 60
total_seconds = total_minutes * 60

print(f"Години:   {total_hours:<10}")
print(f"Хвилини:  {total_minutes:<10}")
print(f"Секунди:  {total_seconds:<10}")