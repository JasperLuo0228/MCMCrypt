import sys, shutil
from optparse import OptionParser
from pathlib import Path
from metropolis_hastings import *
from deciphering_utils    import *
from utils                import az_list

ALPHABET   = az_list()
LETTER_SET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
OTHER_SET  = set(ALPHABET) - LETTER_SET

def apply_map(seq, p_map):
    return ''.join(p_map.get(c, c) for c in seq)

def mapping_accuracy_grouped(pmap, gt_map):

    observed = set(gt_map.keys())
    ov_ok = lt_ok = ot_ok = 0

    for c in observed:
        ok = pmap[c] == gt_map[c]
        ov_ok += ok
        if c in LETTER_SET:
            lt_ok += ok
        else:
            ot_ok += ok

    ov_tot = len(observed)
    lt_tot = len([c for c in observed if c in LETTER_SET])
    ot_tot = len([c for c in observed if c in OTHER_SET])

    res_overall = (ov_ok, ov_tot, ov_ok / ov_tot)

    res_letters = None
    if lt_tot:
        res_letters = (lt_ok, lt_tot, lt_ok / lt_tot)

    res_others  = None
    if ot_tot:
        res_others = (ot_ok, ot_tot, ot_ok / ot_tot)

    return res_overall, res_letters, res_others


def build_gt_map(cipher, plain):
    gt = {}
    for c, p in zip(cipher, plain):
        if c not in gt and c in ALPHABET:
            gt[c] = p
    return gt

def robust_read(path):
    for enc in ("utf-8-sig", "utf-16-le", "utf-16-be", "utf-8", "latin-1"):
        try:
            return Path(path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("robust_read", b"", 0, 0, "all decoders failed")

def main(argv):
    parser = OptionParser()
    parser.add_option("-i","--input", dest="inputfile", help="training corpus")
    parser.add_option("-d","--decode",dest="decode", help="ciphertext file")
    parser.add_option("-r","--reference",dest="ref", help="plaintext reference")
    parser.add_option("-e","--iters",dest="iterations", default=5000,  type="int")
    parser.add_option("-p","--print_every",dest="print_every", default=10000,type="int")
    parser.add_option("-n","--restarts",dest="restarts", default=3,     type="int")
    parser.add_option("-t","--tolerance",dest="tolerance",default=0.02, type="float")
    opts,_ = parser.parse_args(argv)

    if not opts.inputfile or not opts.decode:
        parser.error("-i INPUT and -d DECODE are required")


    train_raw   = robust_read(opts.inputfile)
    train_clean = ''.join(c for c in train_raw if c in ALPHABET)
    Path("._clean_train.txt").write_text(train_clean, encoding="utf-8")
    char_to_ix, ix_to_char, tr, fr = compute_statistics("._clean_train.txt")

    raw_text_str = robust_read(opts.decode).replace("\r\n","\n").replace("\r","\n")
    raw_text   = list(raw_text_str)
    clean_text = [c for c in raw_text if c in ALPHABET]

    reference_raw = robust_read(opts.ref) if opts.ref else None
    reference     = list(reference_raw)[:len(raw_text)] if reference_raw else None
    gt_map        = build_gt_map(clean_text, [c for c in reference if c in ALPHABET]) if reference else {}

    init_state = get_state(clean_text, tr, fr, char_to_ix)

    states, lps = [], []
    for k in range(opts.restarts):
        s, lp, _ = metropolis_hastings(
            init_state,
            proposal_function = propose_a_move,
            log_density       = compute_probability_of_state,
            iters             = opts.iterations,
            print_every       = opts.print_every,
            tolerance         = opts.tolerance,
            pretty_state      = None
        )
        states += s; lps += lp
        print(f"Restart {k+1}/{opts.restarts} done (best logP {max(lp):.0f})")

    ranked = sorted(zip(states,lps), key=lambda x:x[1], reverse=True)

    print("\nBest Guesses:\n")
    for j,(st,lp) in enumerate(ranked[:3], 1):
        pmap    = st["permutation_map"]
        decoded = apply_map(raw_text, pmap)

        metrics = [f"logP {lp:.0f}"]
        if reference:
            (ov,ov_tot,ov_rt),(lt,lt_tot,lt_rt),(ot,ot_tot,ot_rt) = \
                mapping_accuracy_grouped(pmap, gt_map)
            metrics += [
                f"overall {ov}/{ov_tot} ({ov_rt:.2%})",
                f"letters {lt}/{lt_tot} ({lt_rt:.2%})",
                f"others  {ot}/{ot_tot} ({ot_rt:.2%})"
            ]

        print(f"Guess {j}  |  " + " | ".join(metrics) + "\n")
        print(''.join(decoded))
        print('*' * shutil.get_terminal_size().columns)
        
if __name__ == "__main__":
    main(sys.argv[1:])