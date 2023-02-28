
""" This class represents a single word token. 
"""
class Token:
    word = ""

    # Initializer
    def __init__(self, word):
        self.word = word

    
    # Returns the string stored in this token
    def get_word(self):
        return self.word
    

    # Tokens are printed as just their word string
    def __repr__(self):
        return self.word
