# Створіть багаторівневий словник subjects навчальних предметів. Використайте наступні рядки
# для ключів верхнього рівня: &#39;science&#39;, &#39;humanities&#39; і &#39;public&#39;. Зробіть так, щоб ключ
# &#39;science&#39; був ім’ям іншого словника, який має ключі &#39;physics&#39;, &#39;computer science&#39; і
# &#39;biology&#39;. Зробіть так, щоб ключ &#39;physics&#39; посилався на список рядків зі значеннями
#
# &#39;nuclear physics&#39;, &#39;optics&#39; і &#39;thermodynamics&#39;. Решта ключів повинні посилатися на
# порожні словники. Виведіть на екран ключі subjects[&#39;science&#39;] і значення
# subjects[&#39;science&#39;][&#39;physics&#39;].

subjects = {
    'science': {
        'physics': ['nuclear physics', 'optics', 'thermodynamics'],
        'computer science': {},
        'biology': {}
    },
    'humanities': {},
    'public': {}
}

print(subjects['science'].keys())
print(subjects['science']['physics'])