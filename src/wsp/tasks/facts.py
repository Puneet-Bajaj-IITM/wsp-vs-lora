"""Fictional-person facts: knowledge the base model cannot have.

Each invented person has a birth city, a profession and a hobby. Questions carry no context, so the
answer must live in the weights. Train, dev and test ask about the same facts with different templates.
"""
import random

from wsp.tasks.common import make_example

SEED = 23
NUM_PEOPLE = 300
SMALL_NUM_PEOPLE = 60
SYLLABLES = ["ka", "lo", "mir", "ven", "tas", "ru", "del", "nor", "bi", "quen", "sha", "vor", "lei", "dun",
             "pra", "zel", "om", "tri", "gal", "fen"]
FIRST_NAMES = ["Arlen", "Bexa", "Corin", "Dalia", "Emric", "Fenna", "Gorin", "Hesta", "Ilan", "Jorna",
               "Kestrel", "Lunae", "Marrow", "Nessa", "Orrin", "Pella", "Quill", "Rhosyn", "Sorrel", "Tamsin"]
ATTRIBUTES = {
    "city": ["Lisbon", "Osaka", "Nairobi", "Toronto", "Krakow", "Lima", "Hanoi", "Adelaide", "Marseille", "Tbilisi"],
    "profession": ["dentist", "pilot", "carpenter", "librarian", "chemist", "florist", "plumber", "architect",
                   "veterinarian", "journalist"],
    "hobby": ["chess", "pottery", "rock climbing", "birdwatching", "knitting", "surfing", "beekeeping",
              "archery", "origami", "salsa dancing"],
}
TEMPLATES = {
    "city": ["In which city was {n} born?", "{n} was born in which city?", "What is the birthplace of {n}?",
             "Which city is the hometown of {n}?", "Name the city where {n} was born."],
    "profession": ["What does {n} do for a living?", "What is the profession of {n}?", "{n} works as what?",
                   "Which job does {n} have?", "What is the occupation of {n}?"],
    "hobby": ["What is the favourite hobby of {n}?", "Which hobby does {n} enjoy most?",
              "In their free time, what does {n} like to do most?", "What pastime does {n} love?",
              "The main hobby of {n} is which of these?"],
}
SPLIT_TEMPLATES = {"train": (0, 1, 2), "dev": (3,), "test": (4,)}
SYSTEM = "You answer questions about people listed in a private registry. Reply with a single letter."
INSTRUCTION = "Answer with the letter only."


def invent_people(rng: random.Random, num_people: int = NUM_PEOPLE) -> list[str]:
    names: set[str] = set()
    while len(names) < num_people:
        surname = "".join(rng.sample(SYLLABLES, 2) + [rng.choice(SYLLABLES)]).capitalize()
        names.add(f"{rng.choice(FIRST_NAMES)} {surname}")
    return sorted(names)


def build(num_people: int = NUM_PEOPLE) -> list[dict]:
    rng = random.Random(SEED)
    people = invent_people(rng, num_people)
    facts = {(p, a): rng.randrange(len(v)) for p in people for a, v in ATTRIBUTES.items()}
    examples = []
    for (person, attr), value in facts.items():
        for split, template_ids in SPLIT_TEMPLATES.items():
            for t in template_ids:
                examples.append(make_example(
                    system=SYSTEM,
                    body=f"Question: {TEMPLATES[attr][t].format(n=person)}",
                    instruction=INSTRUCTION,
                    item_options=ATTRIBUTES[attr],
                    true_id=value,
                    rng=rng,
                    split=split,
                    column=attr,
                    block=attr,
                    pid=people.index(person),
                    template=t,
                ))
    return examples
