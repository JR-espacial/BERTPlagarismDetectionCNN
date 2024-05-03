import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import os



# Dictionary mapping token names to token values
token_type_dict = {
    # Operators
    'PLUS': 1,
    'MINUS': 2,
    'MULTIPLY': 3,
    'DIVIDE': 4,
    'MODULO': 5,
    'ASSIGN': 6,
    'EQUALS': 7,
    'NOT_EQUALS': 8,
    'LESS_THAN': 9,
    'GREATER_THAN': 10,
    'LESS_THAN_OR_EQUAL': 11,
    'GREATER_THAN_OR_EQUAL': 12,
    'LOGICAL_AND': 13,
    'LOGICAL_OR': 14,
    'LOGICAL_NOT': 15,
    'INCREMENT': 16,
    'DECREMENT': 17,
    # Reserved Words
    'IF': 18,
    'ELSE': 19,
    'WHILE': 20,
    'FOR': 21,
    'SWITCH': 22,
    'CASE': 23,
    'DEFAULT': 24,
    'BREAK': 25,
    'CONTINUE': 26,
    'RETURN': 27,
    # Types of literals
    'INTEGER_LITERAL': 28,
    'LONG_LITERAL': 29,
    'FLOAT_LITERAL': 30,
    'DOUBLE_LITERAL': 31,
    'CHAR_LITERAL': 32,
    'STRING_LITERAL': 33,
    'BOOLEAN_LITERAL': 34,
    'NULL_LITERAL': 35,
    # Types of identifiers
    'VARIABLE_IDENTIFIER': 36,
    'METHOD_IDENTIFIER': 37,
    'CLASS_IDENTIFIER': 38,
    'PACKAGE_IDENTIFIER': 39,
    'INTERFACE_IDENTIFIER': 40,
    'ENUM_IDENTIFIER': 41,
    'ANNOTATION_IDENTIFIER': 42
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
    tokenFile =[]
    lines = java_code.split('\n')
    for line in lines:
        for match in re.finditer(pattern, line):
            token_type = match.lastgroup
            # token_value = match.group() # Uncomment this line if you want to see the value of each token
            token_number = token_type_dict[token_type] # Uncomment this line if you want to see the number of each token
            tokenFile.append(token_number)
    return tokenFile

def getNumTokens():
    return len(token_type_dict)


def fitTokenizer():
    texts =[]
    for file in os.listdir('./fire14-source-code-training-dataset/java/'):
        code = open('./fire14-source-code-training-dataset/java/' + file).read()

        texts.append(code)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return len(tokenizer.index_word)+1, tokenizer


def preprocessData(java_code, tokenizer):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([java_code])
    
    tokenized_code = tokenizer.texts_to_sequences([java_code])
    # print(tokenized_code)
    return tokenized_code

# # Example usage: load file
# java_code = open('./fire14-source-code-training-dataset/java/043.java').read()
# outf = open('output2.txt', 'a')

# tokenLines = lexer(java_code)
# for Line in tokenLines:
#     print("Line:", Line)
#     # Write to file
#     outf.write(' '.join(Line) + '\n')

# outf.close()