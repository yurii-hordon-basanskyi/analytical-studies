# Виконайте візуалізацію структури коду програми. Використайте у візуалізації структури
# елементи кортежу keywords = (&#39;for&#39;, &#39;if&#39;, &#39;else&#39;, &#39;in&#39;, &#39;:&#39;). У процесі
# виведення структури коду на екран, враховуйте відступи рядків від лівого краю, у розрахунку
# один відступ - 4 пропуски. Вигляд структури коду має бути таким:
# for each token in the postfix expression:
#     if the token is a number:
#         print('Convert it to an integer and add it to the end of values')
#     else
#         print('Append the result to the end of values')


keywords = ('for', 'if', 'else', 'in', ':')
indent = '    '

message = (f"{keywords[0]} each token {keywords[3]} the postfix expression{keywords[-1]}"
           f"\n{indent}{keywords[1]} the token is a number{keywords[-1]}"
           f"\n{indent*2}print('Convert it to an integer and add it to the end of values')"
           f"\n{indent}{keywords[2]}"
           f"\n{indent*2}print('Append the result to the end of values')")

print(message)