"""Sign names: the asl-signs vocabulary and human-readable labels for the demo.

``ASL_SIGNS`` is only used by ``export --stub`` before the dataset is
available, so the untrained preview model has real sign names. A trained
export always takes its labels from the prepared cache (``train.csv``).
"""

from __future__ import annotations

ASL_SIGNS: tuple[str, ...] = tuple(
    sorted(
        """
TV after airplane all alligator animal another any apple arm aunt awake backyard bad balloon
bath because bed bedroom bee before beside better bird black blow blue boat book boy brother
brown bug bye callonphone can car carrot cat cereal chair cheek child chin chocolate clean
close closet cloud clown cow cowboy cry cut cute dad dance dirty dog doll donkey down drawer
drink drop dry dryer duck ear elephant empty every eye face fall farm fast feet find fine
finger finish fireman first fish flag flower food for frenchfries frog garbage gift giraffe
girl give glasswindow go goose grandma grandpa grass green gum hair happy hat hate have haveto
head hear helicopter hello hen hesheit hide high home horse hot hungry icecream if into jacket
jeans jump kiss kitty lamp later like lion lips listen look loud mad make man many milk minemy
mitten mom moon morning mouse mouth nap napkin night no noisy nose not now nuts old on open
orange outside owie owl pajamas pen pencil penny person pig pizza please police pool potty
pretend pretty puppy puzzle quiet radio rain read red refrigerator ride room sad same say
scissors see shhh shirt shoe shower sick sleep sleepy smile snack snow stairs stay sticky store
story stuck sun table talk taste thankyou that there think thirsty tiger time tomorrow tongue
tooth toothbrush touch toy tree uncle underwear up vacuum wait wake water wet weus where white
who why will wolf yellow yes yesterday yourself yucky zebra zipper
""".split()  # noqa: SIM905 - a word list reads better as text
    )
)

# Run-together names in the dataset, shown with spaces or as the gloss pair they cover.
_DISPLAY = {
    "callonphone": "call (on phone)",
    "frenchfries": "french fries",
    "glasswindow": "glass / window",
    "haveto": "have to",
    "hesheit": "he / she / it",
    "icecream": "ice cream",
    "minemy": "mine / my",
    "thankyou": "thank you",
    "weus": "we / us",
    "shhh": "shh",
}
# The word to look up in an ASL dictionary.
_LOOKUP = {
    "callonphone": "call",
    "glasswindow": "window",
    "hesheit": "he",
    "minemy": "my",
    "weus": "we",
    "shhh": "quiet",
    "owie": "hurt",
}


def display_name(sign: str) -> str:
    return _DISPLAY.get(sign, sign)


def lookup_word(sign: str) -> str:
    return _LOOKUP.get(sign, display_name(sign))


def humanise_synthetic(name: str) -> str:
    """'circle-open-fist-both' -> 'Circle: open to fist, both hands' (synthetic templates)."""
    parts = name.rstrip("+").split("-")
    both = parts[-1] == "both"
    if both:
        parts = parts[:-1]
    motion, *shapes = parts
    text = f"{motion.capitalize()}: {' to '.join(shapes)} hand"
    return text + (", both hands" if both else "")
