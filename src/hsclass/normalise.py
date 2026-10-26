"""
Text normalisation for cargo descriptions.

The lookup key is a bag of words: uppercase letters-only tokens, packaging / quantity / container
noise removed, duplicates dropped, tokens sorted. So these all map to the same key:

    "1X40HC 24 PALLETS FROZEN BONELESS BEEF"
    "boneless frozen beef - 18.5 MT"
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[A-Z]+")

# Packaging, quantity, container and filler words that say nothing about the commodity.
NOISE = frozenset("""
PALLET PALLETS PLT PLTS CARTON CARTONS CTN CTNS BAG BAGS DRUM DRUMS PKG PKGS PACKAGE PACKAGES
PACKED PACKING PACK PACKS BOX BOXES BALE BALES BULK LOOSE SAID TO CONTAIN CONTAINS CONTAINING
STC FCL LCL CONTAINER CONTAINERS CNTR CNT TOTAL NET GROSS WEIGHT WT KG KGS MT MTS TON TONS TONNE
TONNES NOS PCS PIECES UNITS HC GP FT OF AND THE IN ON WITH FOR FROM BY AS PER ORIGIN MADE FREIGHT
PREPAID COLLECT SHIPPER CONSIGNEE
INDIA BRAZIL VIETNAM CHINA TURKIYE EGYPT GERMANY UAE THAILAND USA SPAIN INDONESIA ITALY MALAYSIA
POLAND UKRAINE
""".split())
# The last two lines are common origin countries: where goods come from does not change the chapter.


def tokens(text: str) -> list[str]:
    """Meaningful tokens in order of appearance (uppercase, letters only, noise removed)."""
    out = []
    for tok in _TOKEN.findall((text or "").upper()):
        if len(tok) < 2 or tok in NOISE:
            continue
        out.append(tok)
    return out


def normalise(text: str) -> str:
    """Bag-of-words lookup key. Empty string when nothing meaningful is left."""
    return " ".join(sorted(set(tokens(text))))
