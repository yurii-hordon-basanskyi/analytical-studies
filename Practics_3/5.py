# Створіть кілька словників, імена яких - це клички домашніх тварин. У кожному словнику збережіть
# інформацію про вид домашнього улюбленця та ім’я власника. Збережіть словники в списку з ім’ям
# pets. Виведіть кілька повідомлень типу Alex is the owner of a pet - a dog..

pet_1 = {'species': 'dog', 'owner': 'Alex'}
pet_2 = {'species': 'cat', 'owner': 'Jay'}
pet_3 = {'species': 'moose', 'owner': 'John'}
pet_4 = {'species': 'aligator', 'owner': 'Olena'}

pets = [pet_1, pet_2, pet_3, pet_4]

for pet in pets:
    print(f"{pet['owner']} is the owner of a pet - a {pet['species']}.")
