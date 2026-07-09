#!/usr/bin/env python3
"""Measure vertical alignment of the three Figure-1 panels in a rendered page.

Usage:
    pdftoppm -gray -r 150 -f 5 -l 5 fig1check.pdf page5   # -> page5-5.pgm
    python3 measure_alignment.py page5-5.pgm [max_row]

Parses the binary PGM (P5), restricts attention to the top `max_row` rows
(default 460 at 150 dpi, i.e. the figure* region), splits the dark content
into three column bands (the three panels), and prints, per band, the
contiguous dark-row intervals (legend block, axis block, xlabel, subcaption).
Alignment is read off by comparing interval boundaries across bands.
"""

import sys


def read_pgm(path):
    with open(path, "rb") as fh:
        data = fh.read()
    # header: P5 <ws> width <ws> height <ws> maxval <single ws> raster
    tokens = []
    i = 0
    while len(tokens) < 4:
        # skip whitespace and comments
        while data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while data[i : i + 1] not in (b"\n", b"\r"):
                i += 1
            continue
        j = i
        while not data[j : j + 1].isspace():
            j += 1
        tokens.append(data[i:j])
        i = j
    i += 1  # single whitespace after maxval
    magic, width, height, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    assert magic == b"P5" and maxval == 255, (magic, maxval)
    raster = data[i : i + width * height]
    assert len(raster) == width * height
    return width, height, raster


def main():
    path = sys.argv[1]
    max_row = int(sys.argv[2]) if len(sys.argv) > 2 else 460
    dark = 128
    width, height, raster = read_pgm(path)
    max_row = min(max_row, height)

    # dark-pixel count per column in the region of interest
    col_counts = [0] * width
    for y in range(max_row):
        row = raster[y * width : (y + 1) * width]
        for x, v in enumerate(row):
            if v < dark:
                col_counts[x] += 1

    # split columns into bands separated by >=15 fully-light columns
    bands = []
    in_band = False
    gap = 0
    start = 0
    for x in range(width):
        if col_counts[x] > 0:
            if not in_band:
                in_band = True
                start = x
            gap = 0
        elif in_band:
            gap += 1
            if gap >= 15:
                bands.append((start, x - gap + 1))
                in_band = False
    if in_band:
        bands.append((start, width - 1))
    bands = [b for b in bands if b[1] - b[0] > 40]  # drop stray marks
    print(f"image {width}x{height}, analysing rows 0..{max_row}")
    print("panel column bands:", bands)

    for bi, (x0, x1) in enumerate(bands):
        # dark rows within band -> contiguous intervals (merge gaps <=2 px)
        rows = []
        for y in range(max_row):
            row = raster[y * width + x0 : y * width + x1 + 1]
            cnt = sum(1 for v in row if v < dark)
            rows.append(cnt)
        intervals = []
        y = 0
        while y < max_row:
            if rows[y] > 0:
                y0 = y
                gap = 0
                widest = 0
                while y < max_row and gap <= 2:
                    if rows[y] > 0:
                        gap = 0
                        widest = max(widest, rows[y])
                    else:
                        gap += 1
                    y += 1
                intervals.append((y0, y - gap - 1, widest))
            else:
                y += 1
        desc = ", ".join(f"[{a}..{b}] (maxdark {w})" for a, b, w in intervals)
        print(f"band {bi} cols {x0}-{x1}: rows {desc}")


if __name__ == "__main__":
    main()
