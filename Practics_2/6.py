keywords = ('for', 'if', 'else', 'in', ':')
indent = '    '

message = (f"{keywords[0]} each token {keywords[3]} the postfix expression{keywords[-1]}"
           f"\n{indent}{keywords[1]} the token is a number{keywords[-1]}"
           f"\n{indent*2}print('Convert it to an integer and add it to the end of values')"
           f"\n{indent}{keywords[2]}"
           f"\n{indent*2}print('Append the result to the end of values')")

print(message)