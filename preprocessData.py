import re

# Dictionary mapping token names to token values
token_type_dict = {
    # Operators
    'PLUS': 100,
    'MINUS': 101,
    'MULTIPLY': 102,
    'DIVIDE': 103,
    'MODULO': 104,
    'ASSIGN': 105,
    'EQUALS': 106,
    'NOT_EQUALS': 107,
    'LESS_THAN': 108,
    'GREATER_THAN': 109,
    'LESS_THAN_OR_EQUAL': 110,
    'GREATER_THAN_OR_EQUAL': 111,
    'LOGICAL_AND': 112,
    'LOGICAL_OR': 113,
    'LOGICAL_NOT': 114,
    'INCREMENT': 115,
    'DECREMENT': 116,
    # Reserved Words
    'IF': 200,
    'ELSE': 201,
    'WHILE': 202,
    'FOR': 203,
    'SWITCH': 204,
    'CASE': 205,
    'DEFAULT': 206,
    'BREAK': 207,
    'CONTINUE': 208,
    'RETURN': 209,
    # Types of literals
    'INTEGER_LITERAL': 300,
    'LONG_LITERAL': 301,
    'FLOAT_LITERAL': 302,
    'DOUBLE_LITERAL': 303,
    'CHAR_LITERAL': 304,
    'STRING_LITERAL': 305,
    'BOOLEAN_LITERAL': 306,
    'NULL_LITERAL': 307,
    # Types of identifiers
    'VARIABLE_IDENTIFIER': 400,
    'METHOD_IDENTIFIER': 401,
    'CLASS_IDENTIFIER': 402,
    'PACKAGE_IDENTIFIER': 403,
    'INTERFACE_IDENTIFIER': 404,
    'ENUM_IDENTIFIER': 405,
    'ANNOTATION_IDENTIFIER': 406
}

# Regular expressions for each token type
patterns = {
    # Operators
    'PLUS': r'\+',
    'MINUS': r'-',
    'MULTIPLY': r'\*',
    'DIVIDE': r'/',
    'MODULO': r'%',
    'ASSIGN': r'=',
    'EQUALS': r'==',
    'NOT_EQUALS': r'!=',
    'LESS_THAN': r'<',
    'GREATER_THAN': r'>',
    'LESS_THAN_OR_EQUAL': r'<=',
    'GREATER_THAN_OR_EQUAL': r'>=',
    'LOGICAL_AND': r'&&',
    'LOGICAL_OR': r'\|\|',
    'LOGICAL_NOT': r'!',
    'INCREMENT': r'\+\+',
    'DECREMENT': r'--',
    # Reserved Words
    'IF': r'if',
    'ELSE': r'else',
    'WHILE': r'while',
    'FOR': r'for',
    'SWITCH': r'switch',
    'CASE': r'case',
    'DEFAULT': r'default',
    'BREAK': r'break',
    'CONTINUE': r'continue',
    'RETURN': r'return',
    # Types of literals
    'INTEGER_LITERAL': r'\b\d+\b',  # Match digits surrounded by word boundaries
    'LONG_LITERAL': r'\b\d+L\b',  # Match digits followed by L surrounded by word boundaries
    'FLOAT_LITERAL': r'\b\d+\.\d+f\b',  # Match floating-point numbers followed by f surrounded by word boundaries
    'DOUBLE_LITERAL': r'\b\d+\.\d+\b',  # Match floating-point numbers surrounded by word boundaries
    'CHAR_LITERAL': r'\'.\'',  # Match single characters surrounded by single quotes
    'STRING_LITERAL': r'\".*?\"',  # Match strings surrounded by double quotes
    'BOOLEAN_LITERAL': r'\b(true|false)\b',  # Match true or false surrounded by word boundaries
    'NULL_LITERAL': r'\bnull\b',  # Match null surrounded by word boundaries
    # Types of identifiers
    'VARIABLE_IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*',
    'METHOD_IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*\(',
    'CLASS_IDENTIFIER': r'[A-Z][a-zA-Z0-9_]*',
    'PACKAGE_IDENTIFIER': r'package [a-z.]+;',
    'INTERFACE_IDENTIFIER': r'interface [a-zA-Z_][a-zA-Z0-9_]*',
    'ENUM_IDENTIFIER': r'enum [a-zA-Z_][a-zA-Z0-9_]*',
    'ANNOTATION_IDENTIFIER': r'@interface [a-zA-Z_][a-zA-Z0-9_]*'
}


# Combine all regex patterns into one
pattern = '|'.join('(?P<%s>%s)' % pair for pair in patterns.items())

def lexer(java_code):
    tokenLines = []
    lines = java_code.split('\n')
    for line in lines:
        tokens = []
        for match in re.finditer(pattern, line):
            token_type = match.lastgroup
            # token_value = match.group() # Uncomment this line if you want to see the value of each token
            token_number = token_type_dict[token_type] # Uncomment this line if you want to see the number of each token
            tokens.append(str(token_number)+ '')
        tokenLines.append(tokens)
    return tokenLines



