#!/usr/bin/python

import sys
import shutil
from optparse import OptionParser
from metropolis_hastings import *
from deciphering_utils import *
from utils import az_list

def main(argv):
    inputfile = None
    decodefile = None
    parser = OptionParser()

    parser.add_option("-i", "--input", dest="inputfile", 
                      help="input file to train the code on")
    parser.add_option("-d", "--decode", dest="decode", 
                      help="file that needs to be decoded")
    parser.add_option("-e", "--iters", dest="iterations", 
                      help="number of iterations to run the algorithm for", default=5000)
    parser.add_option("-t", "--tolerance", dest="tolerance", 
                      help="percentage acceptance tolerance before stopping", default=0.02)
    parser.add_option("-p", "--print_every", dest="print_every", 
                      help="number of steps after which diagnostics should be printed", default=10000)

    (options, args) = parser.parse_args(argv)

    if options.inputfile is None:
        print("Input file is not specified. Type -h for help.")
        sys.exit(2)
    if options.decode is None:
        print("Decoding file is not specified. Type -h for help.")
        sys.exit(2)

    filename = options.inputfile
    char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)

    with open(options.decode, 'r', encoding='utf-8') as f:
        scrambled_text = f.read()

    scrambled_text = " ".join(scrambled_text.replace('\n', ' ')
                                              .replace('\t', ' ')
                                              .replace('\r', ' ')
                                              .split())
    alphabet = az_list()
    scrambled_text = [c for c in scrambled_text if c in alphabet]

    initial_state = get_state(scrambled_text, tr, fr, char_to_ix)
    states = []
    entropies = []

    for i in range(3):
        iters = int(options.iterations)
        print_every = int(options.print_every)
        tolerance = float(options.tolerance)

        state, lps, _ = metropolis_hastings(
            initial_state,
            proposal_function=propose_a_move,
            log_density=compute_probability_of_state,
            iters=iters,
            print_every=print_every,
            tolerance=tolerance,
            pretty_state=pretty_state
        )

        states.extend(state)
        entropies.extend(lps)

    results = list(zip(states, entropies))
    results.sort(key=lambda x: x[1]) 

    print("\nBest Guesses:\n")
    for j in range(1, 4):
        print(f"Guess {j}: \n")
        print(pretty_state(results[-j][0], full=True))
        print('*' * shutil.get_terminal_size().columns)

if __name__ == "__main__":
    main(sys.argv)