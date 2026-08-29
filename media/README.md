# Public Media

This directory contains reviewed publication derivatives only:

- `pipeline/scan-result-cropped.jpg`: authentic detector output cropped above
  the receipt visible in the raw source;
- `evaluation/12-condition-grid.jpg`: one object across the real 3-distance x
  2-lighting x 2-background evaluation matrix.

Both publication files have embedded EXIF, GPS, XMP, and IPTC metadata removed.
Raw inputs remain in the private evidence archive outside Git.

Before adding public media:

1. Remove location metadata and confirm exported files contain no GPS tags.
2. Check every frame for addresses, readable receipts, names, notifications,
   faces, and private account information.
3. Export only the reviewed derivative; keep the original outside Git.

Do not add the private 120-image benchmark or raw voice recordings to this
directory. The committed grid is the approved public sample, not the full corpus.
