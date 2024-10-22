# Створіть словник з ім’ям cities. Використайте назви трьох міст в якості ключів словника.
# Створіть словник з інформацією про кожне місто: включіть в нього країну, в якій розташоване
# місто, приблизну чисельність населення і один цікавий факт про місто. Ключі словника кожного
# міста повинні називатися country, population і fact. Виведіть назву кожного міста і всю
# збережену інформацію про нього.

cities = {
    'Kyiv': {
        'country': 'Ukraine',
        'population': '2.8 million',
        'fact': 'Kyiv is one of the oldest cities in Eastern Europe.'
    },
    'Wroclaw': {
        'country': 'Poland',
        'population': '640,000',
        'fact': 'Wroclaw is known for its beautiful market square and numerous bridges.'
    },
    'Copenhagen': {
        'country': 'Denmark',
        'population': '800,000',
        'fact': 'Copenhagen is famous for the Tivoli Gardens, one of the world’s oldest amusement parks.'
    }
}

for city, data in cities.items():
    print(f"City: {city}")
    for key, value in data.items():
        print(f"  {key.capitalize()}: {value}")