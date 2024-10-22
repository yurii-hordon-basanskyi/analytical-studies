# Cтворіть словник з трьома річками і регіонами, територією яких вони протікають. Одна з можливих
# пар «ключ: значення» - &#39;Amazon&#39;: &#39;South America&#39;. Додайте ще дві пари «річка: регіон» у
# словник. Виведіть повідомлення із назвами річки і регіону - наприклад, «The Amazon runs
# through South America.» для усіх елементів словника, враховуючи те, що у повідомлення у
# відповідні місця підставляються назви річок і територій.

rivers = {
    'Amazon': 'South America',
    'Nile': 'Africa',
    'Dnipro': 'Europe'
}

for river, region in rivers.items():
    print(f"The {river} runs through {region}.")
