"""Select and download Objaverse 3D models for W4 classes the Omniverse library lacks.

Two steps, so human review is recorded and the asset set is reproducible:

    # Workshop host, data root LB=~/nvme2/lawnberrypiserver (containers see $LB/data as /data).
    # 1. review: rank candidates, write candidates.json + contact_<class>.jpg sheets
    python -m workshop.objaverse_select --cache $LB/cache/objaverse \
        --out $LB/data/assets/objaverse --per-class 24
    # 2. after reviewing the sheets, list accepted uids in workshop/objaverse_curation.json
    python -m workshop.objaverse_select --cache $LB/cache/objaverse \
        --out $LB/data/assets/objaverse --curation workshop/objaverse_curation.json

Source: Hugging Face ``allenai/objaverse`` (ODC-By collection; per-object licences
in ``metadata/*.json.gz``). Only CC0 and CC-BY objects are taken: no NC, ND or
SA, so rendered training images carry no extra restriction. ``manifest.json``
keeps uid, class, author and source URL for the attribution CC-BY requires.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import urllib.request
from pathlib import Path

HF = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"
ALLOWED_LICENSES = {"by", "cc0"}
MAX_FACES = 250_000

# LVIS category -> our class (workshop/classes.py names).
LVIS_MAP = {
    "dog": "pet",
    "puppy": "pet",
    "shepherd_dog": "pet",
    "cat": "pet",
    "kitten": "pet",
    "deer": "wildlife",
    "rabbit": "wildlife",
    "squirrel": "wildlife",
    "bird": "wildlife",
    "duck": "wildlife",
    "goose": "wildlife",
    "turtle": "wildlife",
    "snake": "wildlife",
    "garden_hose": "hose",
    "hose": "hose",
}
# Classes with no LVIS category: match the object's name or tags instead.
NAME_RULES = {
    "sprinkler_head": re.compile(r"sprinkler", re.I),
    "root": re.compile(r"\b(tree\s*roots?|exposed\s*roots?|root\s*system|stump)\b", re.I),
}
# Objaverse LVIS labels are noisy: every pick must also name its class.
REQUIRE = {
    "pet": re.compile(
        r"\b(dog|puppy|cat|kitten|kitty|corgi|retriever|labrador|terrier|husky|"
        r"shepherd|beagle|poodle|pug|dachshund|bulldog|collie|spaniel)s?\b",
        re.I,
    ),
    "wildlife": re.compile(
        r"\b(deer|fawn|doe|rabbit|bunny|hare|squirrel|chipmunk|bird|robin|"
        r"sparrow|crow|pigeon|duck|goose|turtle|tortoise|snake|fox|raccoon|"
        r"hedgehog|skunk|possum|opossum)s?\b",
        re.I,
    ),
    "hose": re.compile(r"\bhose\b", re.I),
    "sprinkler_head": re.compile(
        r"\b(lawn|garden|grass|yard|irrigation|pop.?up|rotor|"
        r"oscillating|impact|water)\b",
        re.I,
    ),
    "root": re.compile(r"\b(root|roots|stump)\b", re.I),
}
# Names that are a different object, a scene, or stylised beyond use.
EXCLUDE = re.compile(
    r"\b(fire|extinguisher|ceiling|alarm|gong|bulb|scene|room|level|pack|"
    r"collection|set|lego|minecraft|cartoon|toon|anime|robot|mech|skeleton|"
    r"skull|ax|axe|keychain|figurine|plush|sci-?fi|low.?poly|statue|rubber|"
    r"toy|donut|shop|doge|floppa|cyber\w*|hello|meme|untitled|stylized|"
    r"stylised|cute|chibi|origami|tutorial)\b",
    re.I,
)
REALISM = re.compile(
    r"\b(realistic|photogrammetry|scan|scanned|3d\s*scan|photoscan|pbr|"
    r"game.?ready|lifelike|real)\b",
    re.I,
)


def fetch(url: str, dest: Path) -> Path:
    if not dest.exists() or dest.stat().st_size == 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as fh:
            shutil.copyfileobj(r, fh)
    return dest


def license_key(meta: dict) -> str:
    return (meta.get("license") or "").lower().replace("cc-", "").replace("cc ", "").strip()


def eligible(meta: dict) -> bool:
    return (
        license_key(meta) in ALLOWED_LICENSES
        and not meta.get("isAgeRestricted")
        and 0 < (meta.get("faceCount") or 0) <= MAX_FACES
    )


def score(meta: dict) -> float:
    """Popularity, tripled for scanned/realistic models (training data must look real)."""
    pop = (meta.get("likeCount") or 0) + (meta.get("viewCount") or 0) / 100.0
    tags = " ".join([meta.get("name") or ""] + [t.get("name", "") for t in meta.get("tags") or []])
    return pop * (3.0 if REALISM.search(tags) else 1.0)


def thumbnail_url(meta: dict) -> str | None:
    imgs = (meta.get("thumbnails") or {}).get("images") or []
    imgs = [i for i in imgs if 200 <= (i.get("width") or 0) <= 720] or imgs
    return imgs[0]["url"] if imgs else None


def contact_sheet(entries: list[dict], out: Path) -> None:
    """Grid of source thumbnails with names, for human review of each class."""
    import io

    from PIL import Image, ImageDraw

    cell, cols = 256, 4
    rows = max(1, (len(entries) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 20)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, e in enumerate(entries):
        x, y = (k % cols) * cell, (k // cols) * (cell + 20)
        if e.get("thumbnail"):
            try:
                with urllib.request.urlopen(e["thumbnail"], timeout=60) as r:
                    im = Image.open(io.BytesIO(r.read())).convert("RGB")
                im.thumbnail((cell, cell))
                sheet.paste(im, (x, y))
            except OSError:
                pass
        draw.text((x + 4, y + cell + 4), f"{k} {(e['name'] or '')[:36]}", fill="black")
    sheet.save(out, quality=85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--curation", type=Path, help="JSON {class: [uid, ...]} of reviewed picks")
    a = ap.parse_args()

    def cached_json(rel: str):
        return json.load(gzip.open(fetch(f"{HF}/{rel}", a.cache / rel)))

    lvis = cached_json("lvis-annotations.json.gz")
    paths = cached_json("object-paths.json.gz")
    shards = sorted({p.split("/")[1] for p in paths.values()})
    meta: dict[str, dict] = {}
    for s in shards:
        meta.update(cached_json(f"metadata/{s}.json.gz"))

    candidates: dict[str, list[str]] = {}
    for cat, cls in LVIS_MAP.items():
        candidates.setdefault(cls, []).extend(lvis.get(cat, []))

    def text(m: dict) -> str:
        return " ".join([m.get("name") or ""] + [t.get("name", "") for t in m.get("tags") or []])

    for uid, m in meta.items():
        for cls, rule in NAME_RULES.items():
            if rule.search(text(m)):
                candidates.setdefault(cls, []).append(uid)

    def entry(uid: str, cls: str) -> dict:
        m = meta[uid]
        return {
            "uid": uid,
            "class": cls,
            "name": m.get("name"),
            "license": license_key(m),
            "author": (m.get("user") or {}).get("username"),
            "source": m.get("viewerUrl"),
            "faces": m.get("faceCount"),
            "thumbnail": thumbnail_url(m),
        }

    a.out.mkdir(parents=True, exist_ok=True)
    if a.curation:
        manifest = []
        for cls, uids in json.loads(a.curation.read_text()).items():
            for uid in uids:
                e = entry(uid, cls)
                if not eligible(meta[uid]):
                    raise SystemExit(f"{uid} ({e['name']}) is no longer CC0/CC-BY or eligible")
                e["glb"] = str(
                    fetch(f"{HF}/{paths[uid]}", a.out / "glb" / f"{uid}.glb").relative_to(a.out)
                )
                manifest.append(e)
            print(f"{cls}: {len(uids)} curated", flush=True)
        (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return

    review = []
    for cls, uids in sorted(candidates.items()):
        ok = [
            u
            for u in set(uids)
            if u in meta
            and eligible(meta[u])
            and REQUIRE[cls].search(text(meta[u]))
            and not EXCLUDE.search(meta[u].get("name") or "")
        ]
        picked, names = [], set()
        for u in sorted(ok, key=lambda u: -score(meta[u])):
            key = (meta[u].get("name") or "").strip().lower()
            if key in names:
                continue
            names.add(key)
            picked.append(u)
            if len(picked) == a.per_class:
                break
        entries = [entry(u, cls) for u in picked]
        review.extend(entries)
        contact_sheet(entries, a.out / f"contact_{cls}.jpg")
        print(f"{cls}: {len(picked)} for review from {len(set(uids))} candidates", flush=True)
    (a.out / "candidates.json").write_text(json.dumps(review, indent=2))


if __name__ == "__main__":
    main()
