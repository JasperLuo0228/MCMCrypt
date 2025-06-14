#!/usr/bin/env python
# decode_with_accuracy.py  – letters‑only alphabet & mapping accuracy

import sys, shutil, re
from optparse import OptionParser
from pathlib import Path

from metropolis_hastings_new import metropolis_hastings
from deciphering_utils    import (
    compute_statistics, get_state, scramble_text, pretty_state,
    compute_difference, compute_probability_of_state
)
from deciphering_utils_new import (propose_a_move)

# ─── helpers ──────────────────────────────────────────────────────────
letters_re  = re.compile(r'[a-z ]')      # keep lowercase letters + space
clean_repl  = lambda s: ''.join(ch for ch in s.lower() if letters_re.fullmatch(ch))

def char_accuracy(decoded, reference):
    mism = compute_difference(decoded, reference)
    return 1.0 - mism / len(reference), mism

def word_accuracy(dec_str, ref_str):
    dw, rw = dec_str.split(), ref_str.split()
    L = min(len(dw), len(rw))
    correct = sum(x==y for x,y in zip(dw[:L], rw[:L]))
    return correct/L if L else 0.0

def mapping_accuracy(pmap, gt_map, alphabet):
    wrong = sum(pmap[c] != gt_map.get(c,c) for c in alphabet)
    return 1 - wrong/len(alphabet), wrong

# ─── main ─────────────────────────────────────────────────────────────
def main(argv):
    parser = OptionParser()
    parser.add_option("-i","--input",dest="input",help="training corpus")
    parser.add_option("-d","--decode",dest="decode",help="ciphertext file")
    parser.add_option("-r","--reference",dest="ref",help="plaintext reference")
    parser.add_option("-e","--iters",dest="iters",type="int",default=30000)
    parser.add_option("-p","--print_every",dest="pe",type="int",default=5000)
    parser.add_option("-n","--restarts",dest="rs",type="int",default=3)
    opts,_=parser.parse_args(argv)
    if not opts.input or not opts.decode:
        parser.error("-i INPUT and -d DECODE are required")

    # clean training corpus and build language model
    train_clean = clean_repl(Path(opts.input).read_text())
    Path("._clean_train.txt").write_text(train_clean)         # temp file
    char_to_ix,_,tr,fr = compute_statistics("._clean_train.txt")
    alphabet = list(char_to_ix.keys())                       # 27 symbols

    # clean cipher / reference
    cipher_text = list(clean_repl(Path(opts.decode).read_text()))
    reference   = list(clean_repl(Path(opts.ref).read_text())) if opts.ref else None
    if reference:
        reference = reference[:len(cipher_text)]

    # ground‑truth map if reference provided
    gt_map = {c:r for c,r in zip(cipher_text, reference)} if reference else {}

    state0 = get_state(cipher_text, tr, fr, char_to_ix)
    base_lp= compute_probability_of_state(state0)
    print(f"Identity log‑prob: {base_lp:.2f}\n")

    all_states, all_lp = [], []
    for k in range(opts.rs):
        s,l,_ = metropolis_hastings(state0, propose_a_move,
                                    compute_probability_of_state,
                                    iters=opts.iters, print_every=opts.pe,
                                    tolerance=0.02, pretty_state=None)
        all_states+=s; all_lp+=l
        print(f"Restart {k+1}/{opts.rs} done – best CE {-min(l):.2f}")

    ranked = sorted(zip(all_states,all_lp),key=lambda x:x[1],reverse=True)
    print("\n",shutil.get_terminal_size().columns*"=")
    for j,(st,lp) in enumerate(ranked[:3],1):
        decoded = scramble_text(st["text"], st["permutation_map"])
        metrics=[f"log {lp:.0f}",f"Δ {lp-base_lp:+.0f}"]
        if reference:
            cacc,mis = char_accuracy(decoded,reference)
            macc,merr= mapping_accuracy(st["permutation_map"],gt_map,alphabet)
            metrics+= [f"map {macc:.2%} ({merr}/27)",
                       f"char {cacc:.2%} ({mis} errs)",
                       f"word {word_accuracy(''.join(decoded),''.join(reference)):.2%}"]
        print(f"Guess {j}  ("+" | ".join(metrics)+")\n")
        print(pretty_state(st,full=True))
        print(shutil.get_terminal_size().columns*"-")

# ──────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    main(sys.argv[1:])

