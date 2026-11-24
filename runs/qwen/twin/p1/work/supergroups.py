"""Super-groups of survey items that share an answer style."""
ACTIVE = ("likert", "fc", "econ", "cog_words")
ALL = ACTIVE + ("cog_letters", "other")


def super_group(example, sig):
    s = sig[example["column"]]
    if example["block"] == "Cognitive tests":
        return "cog_letters" if s[0] == "A" else "cog_words"
    if example["block"] == "Economic preferences":
        return "econ"
    if s[0] == "Disagree strongly":
        return "likert"
    if s[0] == "Strongly oppose":
        return "fc"
    return "other"
