"""Option-set groups: items that share the same list of option texts."""
import collections


def option_texts(e):
    return [l[3:] for l in e["user_message"].split("Options:\n")[-1].split("\n")[:5]]


def item_signatures(train):
    txt, n_opt = {}, {}
    for e in train:
        n_opt[e["column"]] = e["n_item_options"]
        for o, t in zip(e["shown_option_ids"], option_texts(e)):
            txt[(e["column"], o)] = t
    return {c: tuple(txt.get((c, o), "?") for o in range(n)) for c, n in n_opt.items()}


def group_log_prior(train, sig):
    counts = collections.defaultdict(collections.Counter)
    for e in train:
        counts[sig[e["column"]]][e["shown_option_ids"][e["label"]]] += 1
    return counts
