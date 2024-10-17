professions = ["Lawyer", "Engineer", "Artist", "Chef", 'Construction worker']

professions.append("Nurse")
print("Після додавання:", professions)

professions.insert(2, "Webmaster")
print("Після вставки:", professions)

professions.remove("Artist")
print("Після видалення:", professions)

professions.pop()
print("Після видалення останнього елементу:", professions)

print("Пошук індексу Chef:", professions.index("Chef"))

print("Кількість входжень Engineer:", professions.count("Engineer"))

professions.sort()
print("Список після сортування:", professions)

professions.reverse()
print("Список після реверсу:", professions)

professions_copy = professions.copy()
print("Копія списку:", professions_copy)

professions.clear()
print("Список після очищення:", professions)
