import random
import string

def generate_mnemonic(len=5):
    letters = string.ascii_letters
    x = "".join(random.sample(letters,len))
    return x