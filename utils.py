import numpy as np
import shutil
import random
from copy import deepcopy
from copy import copy

def az_list():
    """
    Returns all 82 characters in a fixed order.
    """
    ALPHABET = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " ,.;:?!-()\"/\\@#&%$_"
        " "
    )
    return ALPHABET

UPPER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWER = set("abcdefghijklmnopqrstuvwxyz")

def generate_random_permutation_map(chars):
    """
    Generates a randomized character-to-character mapping within groups.

    Characters are grouped and shuffled independently:
    - Uppercase letters are permuted among uppercase
    - Lowercase letters are permuted among lowercase
    - Digits, punctuation, and special symbols are permuted within their own group

    Arguments:
    chars: list of characters (from the fixed 82-character alphabet)

    Returns:
    p_map: dictionary mapping each character to a new character within its group
    """
    upper = [c for c in chars if c in UPPER]
    lower = [c for c in chars if c in LOWER]
    other = [c for c in chars if c not in UPPER and c not in LOWER]

    shuffled_upper = upper[:]; random.shuffle(shuffled_upper)
    shuffled_lower = lower[:]; random.shuffle(shuffled_lower)
    shuffled_other = other[:]; random.shuffle(shuffled_other)

    p_map = {}
    p_map.update(dict(zip(upper, shuffled_upper)))
    p_map.update(dict(zip(lower, shuffled_lower)))
    p_map.update(dict(zip(other, shuffled_other)))

    return p_map

def generate_identity_p_map(chars):
    """
    Generates an identity permutation map for given list of characters
    
    Arguments:
    chars: list of characters
    
    Returns:
    p_map: an identity permutation map
    
    """
    p_map = {}
    for c in chars:
        p_map[c] = c
    
    return p_map
    
def scramble_text(text, p_map):
    """
    Scrambles a text given a permutation map.

    Arguments:
    text: text to scramble, list of characters
    p_map: permutation map to scramble text based upon

    Returns:
    text_2: the scrambled text, with characters replaced using p_map.
            Characters not in p_map are left unchanged.
    """
    text_2 = []
    for c in text:
        text_2.append(p_map.get(c, c))
        
    return text_2
    
def shuffle_text(text, i1, i2):
    """
    Shuffles a text given the index from where to shuffle and
    the upto what we should shuffle
    
    Arguments:
    i1: index from where to start shuffling from
    
    i2: index upto what we should shuffle, excluded.
    """
    
    y = text[i1:i2]
    random.shuffle(y)
    t = copy(text)
    t[i1:i2] = y
    return t
    
def move_one_step(p_map):
    """
    Swaps two characters in the permutation map.

    Arguments:
    p_map: current permutation map (dict)

    Returns:
    p_map_2: a new map with two characters' mappings swapped (deep copy)
    """
    keys = list(p_map.keys())  # use actual map keys, not hardcoded list
    sample = random.sample(keys, 2)

    p_map_2 = deepcopy(p_map)
    p_map_2[sample[1]] = p_map[sample[0]]
    p_map_2[sample[0]] = p_map[sample[1]]

    return p_map_2

def pretty_string(text, full=False):
    """
    Pretty formatted string
    """
    if not full:
        return ''.join(text[1:200]) #+ shutil.get_terminal_size().columns*'-'#'...'
    else:
        return ''.join(text) #+ shutil.get_terminal_size().columns*'-'#'...'
    
def compute_statistics(filename):
    """
    Computes character statistics from a text file using the fixed 82-character alphabet.

    Arguments:
    filename: path to the input text file

    Returns:
    char_to_ix: mapping from character to index (dict)
    ix_to_char: mapping from index to character (dict)
    transition_matrix: smoothed transition probabilities between characters (np.ndarray)
    frequency_statistics: frequency count for each character (np.ndarray)
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    data = " ".join(data.replace("\t", " ").replace("\n", " ").replace("\r", " ").split())
    
    alphabet = az_list() 
    data = [c for c in data if c in alphabet]

    N = len(alphabet)
    char_to_ix = {c: i for i, c in enumerate(alphabet)}
    ix_to_char = {i: c for i, c in enumerate(alphabet)}
    transition_matrix = np.ones((N, N))  
    frequency_statistics = np.zeros(N)

    for i in range(len(data) - 1):
        c1, c2 = data[i], data[i + 1]
        i1, i2 = char_to_ix[c1], char_to_ix[c2]
        transition_matrix[i1, i2] += 1
        frequency_statistics[i1] += 1

    frequency_statistics[char_to_ix[data[-1]]] += 1  # last char
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

    return char_to_ix, ix_to_char, transition_matrix, frequency_statistics