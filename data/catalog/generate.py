#!/usr/bin/env python3
"""Regenerate openngc-positions.csv from the pinned OpenNGC revision.

Run from the repository root: python3 data/catalog/generate.py
Requires only the Python standard library and network access to raw.githubusercontent.com.
"""
import csv
import io
from pathlib import Path
from urllib.request import urlopen

REVISION = "4e2398a263de663d83f7b86be80c368c34335764"
BASE = f"https://raw.githubusercontent.com/mattiaverga/OpenNGC/{REVISION}/database_files"
OUT = Path(__file__).with_name("openngc-positions.csv")
FIELDS = ["designation", "type", "ra_deg", "dec_deg", "major_arcmin", "minor_arcmin", "position_angle_deg", "hubble", "messier", "common_names"]


def coordinate(text, is_ra):
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        first, minute, second = (float(part) for part in parts)
    except ValueError:
        return None
    if not (0 <= minute < 60 and 0 <= second < 60):
        return None
    if is_ra:
        if not 0 <= first < 24:
            return None
        return round(15 * (first + minute / 60 + second / 3600), 8)
    sign = -1 if text.startswith("-") else 1
    degree = abs(first) + minute / 60 + second / 3600
    if degree > 90 or (degree == 90 and (minute or second)):
        return None
    return round(sign * degree, 8)


def extent(text):
    try:
        number = float(text)
    except ValueError:
        return ""
    return f"{number:.2f}" if 0 < number <= 360 * 60 else ""


def generate():
    records = {}
    for filename in ("NGC.csv", "addendum.csv"):
        with urlopen(f"{BASE}/{filename}", timeout=30) as response:
            source = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
            for row in csv.DictReader(source, delimiter=";"):
                if row["Type"] in {"NonEx", "Dup", "Other", "*", "**", "*Ass"}:
                    continue
                ra, dec = coordinate(row["RA"], True), coordinate(row["Dec"], False)
                if ra is None or dec is None:
                    continue
                name = row["Name"].strip()
                if not name or name in records:
                    continue
                major = extent(row["MajAx"])
                minor = extent(row["MinAx"])
                if minor and not major:
                    continue
                if major and not minor:
                    minor = major
                angle = row["PosAng"].strip()
                try:
                    angle = f"{float(angle) % 180:.2f}" if angle else ""
                except ValueError:
                    angle = ""
                messier = row["M"].strip()
                if messier:
                    try:
                        messier = f"M{int(messier)}"
                    except ValueError:
                        messier = ""
                records[name] = [name, row["Type"], f"{ra:.8f}", f"{dec:.8f}", major, minor, angle,
                                 row["Hubble"].strip(), messier, row["Common names"].strip().replace(";", ",")]
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(FIELDS)
        writer.writerows(records[name] for name in sorted(records))
    print(f"{len(records)} positions -> {OUT}")


if __name__ == "__main__":
    generate()
