# Дано список з такими елементами: cities = [&#39;Budapest&#39;, &#39;Rome&#39;, &#39;Istanbul&#39;,
# &#39;Sydney&#39;, &#39;Kyiv&#39;, &#39;Hong Kong&#39;]. Сформуйте з елементів списку повідомлення, у якому
# перед останнім елементом буде вставлено слово and. Наприклад, у нашому випадку,
# повідомлення буде таким: Budapest, Rome, Istanbul, Sydney, Kyiv and Hong
# Kong. Програма має працювати з будь-якими списками, довжина яких є 6.

cities = ['Budapest', 'Rome', 'Istanbul', 'Sydney', 'Kyiv', 'Hong Kong']

message = f"{', '.join(cities[:-1])} and {cities[-1]}"

print(message)