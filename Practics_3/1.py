# Словники Python можуть використовуватися для моделювання «справжнього» словника (назвемо
# його глосарієм). Оберіть кілька термінів з програмування (або із іншої області), які ви знаєте на цей
# момент. Використайте ці слова як ключі глосарію, а їх визначення - як значення. Виведіть кожне
# слово і його визначення у спеціально відформатованому вигляді. Наприклад, ви можете вивести
# слово, потім двокрапка і визначення; або ж слово в одному рядку, а його визначення - з відступом в
# наступному рядку. Використовуйте символ нового рядка (\n) для вставки порожніх рядків між
# парами «слово: визначення» і символ табуляції для встановлення відступів (\t) у вихідних даних.

glossary = {
    'Імператор': 'Божественний лідер людства, який очолює Імперіум та утримує єдність мільярдів планет.',
    'Космічні десантники': 'Елітні війська Імперіуму, генетично вдосконалені суперсолдати, які служать Імператору.',
    'Тираніди': 'Інопланетні біологічні істоти, які поглинають інші форми життя для еволюції та розширення своєї раси.',
    'Хаос': 'Зловісна сила, що виходить із Варпу, яка корумпує розуми та тіла живих істот.',
    'Ельдари': 'Древня раса, яка колись була однією з найбільш могутніх у галактиці, а тепер веде боротьбу за виживання.'
}

for term, definition in glossary.items():
    print(f"{term}:")
    print(f"\t{definition}\n")