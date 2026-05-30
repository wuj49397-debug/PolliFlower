import argparse
import json
import re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def base_stem(stem):
    s = stem

    if ".rf." in s:
        s = s.split(".rf.")[0]

    s = re.sub(r"\s*-\s*副本$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*-\s*副本fz$", "", s, flags=re.IGNORECASE)

    patterns = [
        r"_brightness_[+-]?\d+$",
        r"_brightness_plus\d+$",
        r"_brightness_minus\d+$",
        r"_brightness$",
        r"_dark$",
        r"_rotate_[+-]?\d+$",
        r"_rotate$",
        r"_rotated_[+-]?\d+$",
        r"_rotated[+-]?\d+$",
        r"_rotated$",
        r"_rot[+-]?\d+$",
        r"_rot_[+-]?\d+$",
        r"fz$",
    ]

    changed = True
    while changed:
        changed = False
        for pat in patterns:
            ns = re.sub(pat, "", s, flags=re.IGNORECASE)
            if ns != s:
                s = ns
                changed = True

    return s

def collect_groups(data_dir, split):
    img_dir = data_dir / split / "images"
    groups = {}
    exact_names = set()

    for p in img_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            exact_names.add(p.name)
            groups.setdefault(base_stem(p.stem), []).append(p.name)

    return groups, exact_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data)
    splits = ["train", "val", "test"]

    groups = {}
    exact = {}

    for split in splits:
        groups[split], exact[split] = collect_groups(data_dir, split)

    report = {
        "data": str(data_dir.resolve()),
        "split_images": {s: len(exact[s]) for s in splits},
        "split_base_groups": {s: len(groups[s]) for s in splits},
        "exact_filename_overlap": {},
        "base_group_overlap": {},
    }

    print(f"Checking dataset: {data_dir.resolve()}\n")

    print("Image counts:")
    for s in splits:
        print(f"{s}: images={len(exact[s])}, base_groups={len(groups[s])}")

    print("\nExact filename overlap:")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(exact[a] & exact[b])
        report["exact_filename_overlap"][f"{a}-{b}"] = overlap
        print(f"{a}-{b}: {len(overlap)}")

    print("\nBase-group overlap:")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(set(groups[a]) & set(groups[b]))
        print(f"{a}-{b}: {len(overlap)}")

        items = []
        for g in overlap:
            items.append({
                "base_group": g,
                a: groups[a][g],
                b: groups[b][g],
            })

        report["base_group_overlap"][f"{a}-{b}"] = items

        for item in items[:10]:
            print(" ", item["base_group"])
            print(f"   {a}: {item[a][:5]}")
            print(f"   {b}: {item[b][:5]}")

    out = Path(args.out) if args.out else data_dir / "split_leakage_report_v2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(len(v) for v in report["base_group_overlap"].values())

    print(f"\nSaved report to: {out}")

    if total == 0:
        print("Conclusion: no base-group leakage detected.")
    else:
        print(f"Conclusion: detected {total} base-group overlaps across splits.")

if __name__ == "__main__":
    main()
