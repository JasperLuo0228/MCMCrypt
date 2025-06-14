#!/usr/bin/python
import sys
from optparse import OptionParser
from metropolis_hastings import *
from deciphering_utils import *
from utils import *



def main(argv):
    inputfile = None
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="inputfile", help="file to scramble", default=None)
    (options, args) = parser.parse_args(argv)

    if options.inputfile is None:
        print("File name not specified. Type -h for help.")
        sys.exit(2)

    filename = options.inputfile

    # Load and normalize data
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.read()
    data = " ".join(data.replace("\t", " ").replace("\n", " ").replace("\r", " ").split())

    alphabet = az_list()  # 82 characters
    text = [c for c in data if c in alphabet]

    # Compute statistics and build permutation
    char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)
    p_map = generate_random_permutation_map(list(char_to_ix.keys()))

    # Scramble and print
    scrambled_t = scramble_text(text, p_map)
    with open("scrambled.txt", "w", encoding="utf-8") as f:
        f.write(''.join(scrambled_t))

if __name__ == "__main__":
    main(sys.argv)
