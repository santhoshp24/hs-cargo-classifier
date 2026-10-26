"""
Reference data: Harmonized System (HS) chapters 1-97 and a demo business taxonomy.

* Chapter titles are SHORT paraphrases of the public HS chapter headings, written for readability.
  Check the official WCO publication before relying on exact wording.
* Chapter 77 is reserved for future use in the HS and is never a valid answer.
* The business categories and the "edible" flag below are an invented demo taxonomy, not any
  organisation's real classification scheme. Replace them with your own mapping.
"""
from __future__ import annotations

CHAPTERS: dict[int, str] = {
    1: "Live animals",
    2: "Meat and edible meat offal",
    3: "Fish, crustaceans and molluscs",
    4: "Dairy, eggs, honey",
    5: "Other animal products",
    6: "Live plants, bulbs, cut flowers",
    7: "Edible vegetables, roots and tubers",
    8: "Edible fruit and nuts",
    9: "Coffee, tea, spices",
    10: "Cereals",
    11: "Milling products, malt, starches",
    12: "Oil seeds, grains, seeds, fodder plants",
    13: "Lac, gums, resins, plant extracts",
    14: "Vegetable plaiting materials",
    15: "Animal and vegetable fats and oils",
    16: "Preparations of meat and fish",
    17: "Sugars and sugar confectionery",
    18: "Cocoa and cocoa preparations",
    19: "Cereal, flour and pastry preparations",
    20: "Vegetable and fruit preparations",
    21: "Miscellaneous edible preparations",
    22: "Beverages, spirits and vinegar",
    23: "Food industry residues, animal feed",
    24: "Tobacco and nicotine products",
    25: "Salt, sulphur, earths, stone, cement",
    26: "Ores, slag and ash",
    27: "Mineral fuels, oils and waxes",
    28: "Inorganic chemicals",
    29: "Organic chemicals",
    30: "Pharmaceutical products",
    31: "Fertilisers",
    32: "Dyes, pigments, paints, inks",
    33: "Essential oils, perfumery, cosmetics",
    34: "Soap, detergents, candles, polishes",
    35: "Albuminoids, modified starches, glues, enzymes",
    36: "Explosives, pyrotechnics, matches",
    37: "Photographic goods",
    38: "Miscellaneous chemical products",
    39: "Plastics and articles thereof",
    40: "Rubber and articles thereof",
    41: "Raw hides, skins and leather",
    42: "Leather articles, handbags, travel goods",
    43: "Furskins and artificial fur",
    44: "Wood and articles of wood",
    45: "Cork and articles of cork",
    46: "Straw and plaiting manufactures, basketware",
    47: "Wood pulp, recovered paper",
    48: "Paper and paperboard",
    49: "Printed books and printed matter",
    50: "Silk",
    51: "Wool and animal hair",
    52: "Cotton",
    53: "Other vegetable textile fibres",
    54: "Man-made filaments",
    55: "Man-made staple fibres",
    56: "Wadding, felt, nonwovens, cordage",
    57: "Carpets and floor coverings",
    58: "Special woven fabrics, lace, embroidery",
    59: "Coated and laminated textile fabrics",
    60: "Knitted or crocheted fabrics",
    61: "Apparel, knitted or crocheted",
    62: "Apparel, not knitted or crocheted",
    63: "Other made-up textile articles, worn clothing",
    64: "Footwear",
    65: "Headgear",
    66: "Umbrellas, walking sticks, whips",
    67: "Feathers, artificial flowers, human hair articles",
    68: "Articles of stone, plaster, cement, mica",
    69: "Ceramic products",
    70: "Glass and glassware",
    71: "Pearls, gems, precious metals, jewellery",
    72: "Iron and steel",
    73: "Articles of iron or steel",
    74: "Copper and articles thereof",
    75: "Nickel and articles thereof",
    76: "Aluminium and articles thereof",
    78: "Lead and articles thereof",
    79: "Zinc and articles thereof",
    80: "Tin and articles thereof",
    81: "Other base metals, cermets",
    82: "Tools, cutlery of base metal",
    83: "Miscellaneous articles of base metal",
    84: "Machinery, boilers, mechanical appliances",
    85: "Electrical machinery and equipment",
    86: "Railway rolling stock and track fittings",
    87: "Vehicles (not railway) and parts",
    88: "Aircraft and spacecraft",
    89: "Ships, boats and floating structures",
    90: "Optical, measuring, medical instruments",
    91: "Clocks and watches",
    92: "Musical instruments",
    93: "Arms and ammunition",
    94: "Furniture, bedding, lighting, prefab buildings",
    95: "Toys, games and sports equipment",
    96: "Miscellaneous manufactured articles",
    97: "Works of art and antiques",
}

RESERVED_CHAPTERS = {77}
VALID_CHAPTERS = frozenset(CHAPTERS)

# Demo taxonomy: category name -> chapters it contains.
_CATEGORY_CHAPTERS: dict[str, list[int]] = {
    "Food & Beverages": [2, 3, 4, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22],
    "Agri Raw Materials & Feed": [1, 5, 6, 12, 13, 14, 23, 24],
    "Minerals, Ores & Fuels": [25, 26, 27],
    "Chemicals & Fertilisers": list(range(28, 39)),
    "Plastics & Rubber": [39, 40],
    "Wood, Paper & Pulp": [44, 45, 46, 47, 48, 49],
    "Textiles, Apparel & Leather": [41, 42, 43] + list(range(50, 68)),
    "Stone, Ceramics, Glass & Gems": [68, 69, 70, 71],
    "Metals & Metal Articles": list(range(72, 77)) + list(range(78, 84)),
    "Machinery & Electronics": [84, 85, 90, 91, 92],
    "Vehicles & Transport Equipment": [86, 87, 88, 89],
    "Furniture, Toys & Other Goods": [93, 94, 95, 96, 97],
}

CATEGORY_BY_CHAPTER: dict[int, str] = {
    ch: cat for cat, chs in _CATEGORY_CHAPTERS.items() for ch in chs
}

EDIBLE_CATEGORY = "Food & Beverages"


def chapter_title(chapter: int) -> str:
    return CHAPTERS.get(chapter, "")


def category_for(chapter: int) -> str:
    return CATEGORY_BY_CHAPTER.get(chapter, "Unclassified")


def is_edible(chapter: int) -> bool:
    """Demo rule: a chapter is 'edible' when it belongs to the Food & Beverages category."""
    return category_for(chapter) == EDIBLE_CATEGORY


def chapter_prompt_table() -> str:
    """Compact 'NN title' lines used inside LLM prompts."""
    return "\n".join(f"{ch:02d} {title}" for ch, title in sorted(CHAPTERS.items()))


def _self_check() -> None:
    missing = VALID_CHAPTERS - set(CATEGORY_BY_CHAPTER)
    extra = set(CATEGORY_BY_CHAPTER) - VALID_CHAPTERS
    if missing or extra:
        raise AssertionError(f"taxonomy mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    if len(VALID_CHAPTERS) != 96:  # 97 chapters minus reserved chapter 77
        raise AssertionError(f"expected 96 valid chapters, got {len(VALID_CHAPTERS)}")


_self_check()
