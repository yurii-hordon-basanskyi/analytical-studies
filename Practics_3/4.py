# Створіть англо-німецький словник, який називається e2g, і виведіть його на екран. Слова для
# словника: stork / storch, hawk / falke, woodpecker / specht і owl / eule. Виведіть
# німецький варіант слова owl. Додайте у словник, на ваш вибір, ще два слова та їхній переклад.
# Виведіть окремо: словник; ключі і значення словника у вигляді списків.

e2g = {
    'stork': 'storch',
    'hawk': 'falke',
    'woodpecker': 'specht',
    'owl': 'eule'
}

print(e2g)

print(f"Переклад слова 'owl' німецькою: {e2g['owl']}")

e2g['sparrow'] = 'spatz'
e2g['eagle'] = 'adler'

print(e2g)

print(f"Ключі: {list(e2g.keys())}")
print(f"Значення: {list(e2g.values())}")