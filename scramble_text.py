#!/usr/bin/python

import sys
from optparse import OptionParser
from utils import *
from deciphering_utils import *
from metropolis_hastings import *

def main(argv):
    inputfile = None
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="inputfile", help="file to scramble", default=None)
    (options, args) = parser.parse_args(argv)

    if options.inputfile is None:
        print("File name not specified. Type -h for help.")
        sys.exit(2)

    filename = options.inputfile

    with open(filename, 'r', encoding='utf-8') as f:
        original_text = f.read()

    alphabet = az_list() 
    char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)
    p_map = generate_random_permutation_map(list(char_to_ix.keys()))

    scrambled_t = []
    for c in original_text:
        if c in alphabet:
            scrambled_t.append(p_map[c])
        else:
            scrambled_t.append(c)  

    with open("scrambled.txt", "w", encoding="utf-8") as f:
        f.write(''.join(scrambled_t))

if __name__ == "__main__":
    main(sys.argv)
