"""
Synthetic cargo data. Everything here is invented: product names are generic commodity phrases,
container numbers use a made-up owner prefix, dates are random. Ground truth (true_hs_chapter) is
included so accuracy can be measured. It is seeded, so the same seed gives the same files.
"""
from __future__ import annotations

import csv
import datetime as dt
import random
from pathlib import Path

from .hs_reference import CATEGORY_BY_CHAPTER, CHAPTERS, VALID_CHAPTERS, is_edible

# (chapter, [clean commodity phrases]).  Phrases are the "clean" form a curated lookup would hold.
PRODUCTS: list[tuple[int, list[str]]] = [
    (2, ["FROZEN BONELESS BEEF", "CHILLED MUTTON CARCASSES", "FROZEN CHICKEN LEG QUARTERS", "FROZEN BUFFALO MEAT BONELESS"]),
    (3, ["FROZEN SHRIMP HEAD ON", "CHILLED SALMON FILLETS", "FROZEN TUNA LOINS", "DRIED SALTED FISH"]),
    (4, ["FULL CREAM MILK POWDER", "UNSALTED BUTTER", "CHEDDAR CHEESE BLOCKS", "NATURAL HONEY", "FRESH TABLE EGGS"]),
    (7, ["FRESH ONIONS", "TABLE POTATOES", "FROZEN GREEN PEAS", "DRIED CHICKPEAS", "FRESH CARROTS"]),
    (8, ["FRESH CAVENDISH BANANAS", "FRESH ORANGES", "KHALAS DATES", "RAW CASHEW NUTS", "GALA APPLES", "CALIFORNIA ALMONDS"]),
    (9, ["GREEN ARABICA COFFEE BEANS", "BLACK TEA CTC", "WHOLE BLACK PEPPER", "GREEN CARDAMOM", "TURMERIC FINGERS", "CUMIN SEEDS"]),
    (10, ["MILLING WHEAT", "BASMATI RICE", "YELLOW MAIZE FEED GRADE", "MALTING BARLEY"]),
    (11, ["WHEAT FLOUR BAKERY GRADE", "MAIZE STARCH"]),
    (12, ["NON GMO SOYBEANS", "HULLED SESAME SEEDS", "SUNFLOWER SEEDS", "GROUNDNUT KERNELS"]),
    (15, ["CRUDE PALM OIL", "REFINED SUNFLOWER OIL", "EXTRA VIRGIN OLIVE OIL", "CRUDE SOYBEAN OIL"]),
    (16, ["CANNED TUNA IN OIL", "CANNED SARDINES", "PORK SAUSAGES"]),
    (17, ["REFINED WHITE SUGAR ICUMSA 45", "RAW CANE SUGAR", "ASSORTED CONFECTIONERY"]),
    (18, ["COCOA BEANS", "COCOA POWDER", "CHOCOLATE BARS"]),
    (19, ["DURUM PASTA SPAGHETTI", "ASSORTED BISCUITS", "BREAKFAST CEREALS", "INSTANT NOODLES"]),
    (20, ["TOMATO PASTE IN DRUMS", "FRUIT JUICE CONCENTRATE", "CANNED PINEAPPLE SLICES", "MIXED PICKLES"]),
    (21, ["INSTANT COFFEE", "SOY SAUCE", "ICE CREAM", "TOMATO KETCHUP"]),
    (22, ["BOTTLED MINERAL WATER", "SOFT DRINKS IN CANS", "ENERGY DRINKS", "WHITE VINEGAR"]),
    (23, ["SOYBEAN MEAL", "WHEAT BRAN PELLETS", "DISTILLERS DRIED GRAINS DDGS", "DOG FOOD DRY"]),
    (24, ["UNMANUFACTURED TOBACCO LEAF", "CIGARETTES"]),
    (25, ["ORDINARY PORTLAND CEMENT", "NATURAL GYPSUM", "INDUSTRIAL SALT", "SILICA SAND", "GRANITE BLOCKS"]),
    (26, ["IRON ORE FINES", "COPPER CONCENTRATE", "BAUXITE"]),
    (27, ["STEAM COAL", "LIQUEFIED PETROLEUM GAS LPG", "BITUMEN 60/70", "GASOIL 10PPM", "CALCINED PETROLEUM COKE", "LIGHT NAPHTHA"]),
    (28, ["CAUSTIC SODA FLAKES", "SULPHURIC ACID", "SODA ASH DENSE"]),
    (29, ["METHANOL", "MONO ETHYLENE GLYCOL", "ACETONE"]),
    (30, ["PHARMACEUTICAL TABLETS", "MEDICAMENTS IN RETAIL PACKS", "VACCINES TEMPERATURE CONTROLLED"]),
    (31, ["UREA FERTILIZER PRILLED", "NPK FERTILIZER", "DAP FERTILIZER"]),
    (32, ["ACRYLIC PAINT", "TITANIUM DIOXIDE PIGMENT", "PRINTING INK"]),
    (33, ["PERFUME", "HAIR SHAMPOO", "SKIN CREAM COSMETICS"]),
    (34, ["LAUNDRY DETERGENT POWDER", "TOILET SOAP BARS", "WAX CANDLES"]),
    (38, ["SURFACE DISINFECTANT", "ENGINE ANTIFREEZE COOLANT", "AGRICULTURAL PESTICIDE"]),
    (39, ["POLYETHYLENE GRANULES HDPE", "PVC PIPES", "PLASTIC CARRIER BAGS", "PET RESIN BOTTLE GRADE"]),
    (40, ["NEW PNEUMATIC TYRES", "RUBBER GLOVES", "NATURAL RUBBER SHEETS"]),
    (41, ["FINISHED COW LEATHER"]),
    (42, ["LEATHER HANDBAGS", "TRAVEL SUITCASES"]),
    (44, ["SAWN PINE TIMBER", "BIRCH PLYWOOD", "WOODEN PALLETS", "WOOD PELLETS"]),
    (47, ["BLEACHED WOOD PULP", "WASTE PAPER OCC"]),
    (48, ["KRAFT PAPER ROLLS", "CORRUGATED CARTONS FLAT PACKED", "TISSUE PAPER JUMBO ROLLS"]),
    (49, ["PRINTED BOOKS", "ADVERTISING BROCHURES"]),
    (52, ["COTTON YARN", "COTTON WOVEN FABRIC"]),
    (54, ["POLYESTER FILAMENT YARN"]),
    (55, ["POLYESTER STAPLE FIBRE"]),
    (57, ["MACHINE MADE CARPETS"]),
    (61, ["KNITTED COTTON T SHIRTS", "KNITTED WOOL SWEATERS"]),
    (62, ["MENS WOVEN TROUSERS", "LADIES DRESSES"]),
    (63, ["COTTON BED LINEN", "BATH TOWELS", "USED CLOTHING BALES"]),
    (64, ["SPORTS SHOES", "LEATHER FOOTWEAR"]),
    (68, ["POLISHED MARBLE TILES"]),
    (69, ["CERAMIC FLOOR TILES", "PORCELAIN DINNERWARE", "SANITARY WARE"]),
    (70, ["FLOAT GLASS SHEETS", "GLASS BOTTLES", "TABLE GLASSWARE"]),
    (71, ["GOLD JEWELLERY", "IMITATION JEWELLERY"]),
    (72, ["HOT ROLLED STEEL COILS", "STEEL REBAR", "STAINLESS STEEL SHEETS", "PIG IRON"]),
    (73, ["SEAMLESS STEEL PIPES", "STEEL WIRE ROPE", "STEEL BOLTS AND NUTS FASTENERS", "FABRICATED STEEL STRUCTURES"]),
    (74, ["COPPER CATHODES", "COPPER WIRE"]),
    (76, ["ALUMINIUM INGOTS", "ALUMINIUM EXTRUSIONS", "ALUMINIUM FOIL"]),
    (79, ["ZINC INGOTS"]),
    (82, ["HAND TOOLS SPANNERS", "STAINLESS CUTLERY SETS"]),
    (83, ["PADLOCKS"]),
    (84, ["CENTRIFUGAL PUMPS", "AIR CONDITIONERS SPLIT UNITS", "HYDRAULIC EXCAVATOR", "MOBILE CRANE",
          "LAPTOP COMPUTERS", "DOMESTIC REFRIGERATORS", "AIR COMPRESSORS"]),
    (85, ["MOBILE PHONES", "ELECTRIC MOTORS", "LED TELEVISIONS", "LITHIUM BATTERIES", "SOLAR PANELS",
          "DIESEL GENERATING SETS", "ELECTRIC CABLES"]),
    (86, ["RAILWAY WAGONS"]),
    (87, ["PASSENGER CARS", "HEAVY TRUCKS", "MOTORCYCLES", "AGRICULTURAL TRACTORS", "BRAKE PADS"]),
    (88, ["AIRCRAFT PARTS"]),
    (89, ["TUG BOAT", "INFLATABLE BOATS"]),
    (90, ["DISPOSABLE MEDICAL SYRINGES", "SURGICAL INSTRUMENTS", "MEASURING INSTRUMENTS"]),
    (91, ["WRIST WATCHES"]),
    (94, ["OFFICE FURNITURE", "FOAM MATTRESSES", "LED LIGHTING FIXTURES", "PREFAB CABINS"]),
    (95, ["ASSORTED TOYS", "GYM SPORTS EQUIPMENT", "FOOTBALLS"]),
    (96, ["BALL PENS", "PAINT BRUSHES", "ZIPPERS"]),
    (97, ["ORIGINAL PAINTINGS"]),
]

PACKS = ["PALLETS", "CARTONS", "BAGS", "DRUMS", "BALES", "PACKAGES", "BOXES"]
CONTAINERS = ["1X40HC", "1X20GP", "2X40FT", "1X40 HC", "20FT", "40HC", "3X20GP"]
ORIGINS = ["INDIA", "BRAZIL", "VIETNAM", "CHINA", "TURKIYE", "EGYPT", "GERMANY", "UAE", "THAILAND",
           "USA", "SPAIN", "INDONESIA", "ITALY", "MALAYSIA", "POLAND", "UKRAINE"]
PREFIXES = ["", "", "", "SAID TO CONTAIN", "FCL", "STC"]


def _typo(text: str, rng: random.Random) -> str:
    words = text.split()
    cand = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not cand:
        return text
    i = rng.choice(cand)
    w = list(words[i])
    j = rng.randrange(1, len(w) - 2)
    w[j], w[j + 1] = w[j + 1], w[j]
    words[i] = "".join(w)
    return " ".join(words)


def _variant(phrase: str, rng: random.Random) -> str:
    pack, cont = rng.choice(PACKS), rng.choice(CONTAINERS)
    n, mt = rng.randint(2, 60), round(rng.uniform(5, 28), 1)
    pre, origin = rng.choice(PREFIXES), rng.choice(ORIGINS)
    templates = [
        f"{cont} {pre} {n} {pack} {phrase}",
        f"{phrase} - {n} {pack}",
        f"{phrase} ORIGIN {origin}",
        f"{pre} {phrase} IN {pack}",
        f"{mt} MT {phrase}",
        f"{phrase}",
        f"{n} {pack} OF {phrase} PACKED IN {cont}",
    ]
    text = " ".join(rng.choice(templates).split())
    if rng.random() < 0.05:
        text = _typo(text, rng)
    roll = rng.random()
    if roll < 0.15:
        text = text.lower()
    elif roll < 0.25:
        text = text.title()
    return text


def generate(out_dir: Path, rows: int = 2000, seed: int = 42, curated_share: float = 0.30) -> dict:
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # One entry per phrase, with a heavy-tailed popularity so some goods repeat a lot (as in real data).
    catalogue = []
    for chapter, phrases in PRODUCTS:
        for phrase in phrases:
            variants = [_variant(phrase, rng) for _ in range(rng.randint(2, 6))]
            catalogue.append({"chapter": chapter, "phrase": phrase, "variants": variants,
                              "weight": rng.paretovariate(1.3)})
    weights = [c["weight"] for c in catalogue]

    start, span = dt.date(2025, 1, 1), 640
    records = []
    for i in range(1, rows + 1):
        item = rng.choices(catalogue, weights=weights, k=1)[0]
        cont = rng.choice(CONTAINERS)
        records.append({
            "record_id": f"CR{i:06d}",
            "container_no": f"DEMU{rng.randint(0, 9999999):07d}",  # made-up owner prefix
            "arrival_date": (start + dt.timedelta(days=rng.randrange(span))).isoformat(),
            "teu": 2 if "40" in cont else 1,
            "cargo_description": rng.choice(item["variants"]),
            "true_hs_chapter": item["chapter"],
        })

    with open(out_dir / "sample_cargo.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    # Curated seed: clean phrases for a share of the catalogue, as a human-maintained table would be.
    seeded = [c for c in catalogue if rng.random() < curated_share]
    with open(out_dir / "curated_lookup.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["description", "hs_chapter"])
        for c in seeded:
            w.writerow([c["phrase"], c["chapter"]])

    write_chapter_dimension(out_dir / "dim_hs_chapter.csv")
    return {"rows": len(records), "products": len(catalogue), "curated_seed": len(seeded),
            "chapters_covered": len({c["chapter"] for c in catalogue})}


def write_chapter_dimension(path: Path) -> None:
    """Chapter dimension for a Power BI star schema (join classified.csv[hs_chapter] to it)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hs_chapter", "chapter_title", "cargo_category", "edible"])
        for ch in sorted(VALID_CHAPTERS):
            w.writerow([ch, CHAPTERS[ch], CATEGORY_BY_CHAPTER[ch], "Edible" if is_edible(ch) else "Non-Edible"])
