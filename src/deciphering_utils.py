import numpy as np
import random
from utils import *

def compute_log_probability(text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the 
    charcter c from permutation_map[c]), given the text statistics
    
    Note: This is quite slow, as it goes through the whole text to compute the probability,
    if you need to compute the probabilities frequently, see compute_log_probability_by_counts.
    
    Arguments:
    text: text, list of characters
    
    permutation_map[c]: gives the character to replace 'c' by
    
    char_to_ix: characters to index mapping
    
    frequency_statistics: frequency of character i is stored in frequency_statistics[i]
    
    transition_matrix: probability of j following i
    
    Returns:
    p: log likelihood of the given text
    """
    t = text
    p_map = permutation_map
    cix = char_to_ix
    fr = frequency_statistics
    tm = transition_matrix
    
    i0 = cix[p_map[t[0]]]
    p = np.log(fr[i0])
    i = 0
    while i < len(t)-1:
        subst = p_map[t[i+1]]
        i1 = cix[subst]
        p += np.log(tm[i0, i1])
        i0 = i1
        i += 1
        
    return p

def compute_transition_counts(text, char_to_ix):
    """
    Computes transition counts for a given text, useful to compute if you want to compute 
    the probabilities again and again, using compute_log_probability_by_counts.
    
    Arguments:
    text: Text as a list of characters
    
    char_to_ix: character to index mapping
    
    Returns:
    transition_counts: transition_counts[i, j] gives number of times character j follows i
    """
    N = len(char_to_ix)
    transition_counts = np.zeros((N, N))
    c1 = text[0]
    i = 0
    while i < len(text)-1:
        c2 = text[i+1]
        transition_counts[char_to_ix[c1],char_to_ix[c2]] += 1
        c1 = c2
        i += 1
    
    return transition_counts

def compute_log_probability_by_counts(transition_counts, text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map.
    """

    eps = 1e-8  # small constant to avoid log(0)

    # Map first character
    first_char = permutation_map.get(text[0], text[0])
    c0 = char_to_ix.get(first_char, None)
    if c0 is None:
        return -np.inf  # unknown character — return lowest possible log prob
    frequency_statistics = np.clip(frequency_statistics, 1e-8, None)
    p = np.log(frequency_statistics[c0])

    # Build remapped index list
    try:
        indices = [char_to_ix[permutation_map[c]] for c in char_to_ix]
    except KeyError:
        return -np.inf  # bad permutation

    # Log transition matrix with epsilon
    log_tm = np.log(transition_matrix + eps)
    log_tm_sub = log_tm[indices, :][:, indices]

    p += np.sum(transition_counts * log_tm_sub)

    return p


def compute_difference(text_1, text_2):
    """
    Compute the number of times to text differ in character at same positions
    
    Arguments:
    
    text_1: first text list of characters
    text_2: second text, should have same length as text_1
    
    Returns
    cnt: number of times the texts differ in character at same positions
    """
    
    cnt = 0
    for x, y in zip(text_1, text_2):
        if y != x:
            cnt += 1
            
    return cnt

def get_state(text, transition_matrix, frequency_statistics, char_to_ix):
    """
    Generates a default state of given text statistics
    
    Arguments:
    pretty obvious
    
    Returns:
    state: A state that can be used along with,
           compute_probability_of_state, propose_a_move,
           and pretty_state for metropolis_hastings
    
    """
    transition_counts = compute_transition_counts(text, char_to_ix)
    p_map = generate_identity_p_map(char_to_ix.keys())
    
    state = {"text" : text, "transition_matrix" : transition_matrix, 
             "frequency_statistics" : frequency_statistics, "char_to_ix" : char_to_ix,
            "permutation_map" : p_map, "transition_counts" : transition_counts}
    
    return state

def compute_probability_of_state(state):
    """
    Computes the probability of given state using compute_log_probability_by_counts
    """
    
    p = compute_log_probability_by_counts(state["transition_counts"], state["text"], state["permutation_map"], 
                                          state["char_to_ix"], state["frequency_statistics"], state["transition_matrix"])
    
    return p

import numpy as np

# Character groups (must match your 82-symbol ALPHABET)
LETTER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
PUNCT_SET  = set("0123456789 ,.;:?!-()\"/\\@#&%$_ ")   # 19 symbols + space

def propose_a_move(state, eps: float = 1e-6,
                       p_letter=0.6, p_punct=0.3):
    """Frequency-weighted, symmetric *mixture* proposal."""
    rng   = np.random.default_rng()
    u     = rng.random()
    p_map = dict(state["permutation_map"])
    freqs = state["frequency_statistics"]
    char_ix = state["char_to_ix"]

    def _weighted_choice(pool):
        pool_arr = np.array(list(pool))
        ix_arr   = [char_ix[c] for c in pool_arr]
        w        = np.abs(freqs[ix_arr] - freqs.mean()) + eps
        w        = w / w.sum()
        return rng.choice(pool_arr, p=w)

    # choose the pool to sample from
    if u < p_letter:
        pool = LETTER_SET
    elif u < p_letter + p_punct:
        pool = PUNCT_SET
    else:
        pool = p_map.keys()      # fall-back to “swap any”

    # draw two distinct symbols
    while True:
        c1 = _weighted_choice(pool)
        c2 = _weighted_choice(pool)
        if c1 != c2:
            break

    # swap
    p_map[c1], p_map[c2] = p_map[c2], p_map[c1]

    new_state = dict(state)
    new_state["permutation_map"] = p_map
    return new_state


def pretty_state(state, full=True):
    """
    Returns the state in a pretty format
    """
    if not full:
        return pretty_string(scramble_text(state["text"][1:200], state["permutation_map"]), full)
    else:
        return pretty_string(scramble_text(state["text"], state["permutation_map"]), full)