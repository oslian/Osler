# Anatomy motion assets

`PROVENANCE.json` is the record of record. It lists every asset actually shipped
here with its own entry URL, title, entry id, download date, individual licence
and verbatim rights statement, the local filename, and exactly what was changed.
Read it before adding, replacing or removing anything in this directory.

Two rules govern this directory:

1. **Licences are per asset, never per collection.** BioArt Source interleaves
   CC BY items from the Human Reference Atlas among public-domain ones, so the
   per-entry rights field is what was checked, not the site's general terms.
2. **A credit line is kept wherever a source recommends one**, including where
   the licence does not compel it. Nothing here is recorded as "no attribution".

## What is shipped

| Asset | Source | Licence | Credit kept |
|---|---|---|---|
| `body-ladyofhats-handfix.svg` + its `.back` / `.front` / `.vessels` display variants | *Arterial System en.svg*, Wikimedia Commons | Public domain, verified through the Commons API | LadyofHats (Mariana Ruiz Villarreal) |
| `organs/*.svg` — 8 organ illustrations | NIH BioArt Source, NIAID Visual & Medical Arts | Public domain, per entry | Ryan Kissinger · Courtesy of NIAID |
| `organs/brain-anterior.svg` | Drawn for this project | **Oslian original derivative artwork** | Oslian |
| `organ-layout.json` — placement manifest | Authored for this project | Oslian project-owned | Oslian |

`organs/SOURCES.json` carries the per-organ record, including each asset's view
(`anterior` / `lateral`) and whether it is actually drawn. `organs/brain.svg` is
a lateral view: it is shipped with its licence but is not drawn into this
front-facing figure.

## Placement

The figure keeps the artist's own coordinates in one canonical 550 × 830 frame;
the renderer applies no correction transform. Organs are placed by
`organ-layout.json` using translation and uniform scale only — a paired organ is
reflected about its own centre line — so no source artwork is ever warped.

## What was drawn rather than sourced, and why

- **Anterior brain** — the only brain in the BioArt catalogue is a lateral view,
  which cannot be drawn into a front-facing figure. `organs/brain-anterior.svg`
  was authored for this project instead.

Thyroid, skeletal muscle and bone marrow are on the canonical organ list but no
surveyed atlas carries them as usable artwork. They currently ship with no
illustration; the figure marks their region and the UI reports them as organs
without an asset.

## Excluded sources

| Source | Why not |
|---|---|
| Z-Anatomy | Ships a CC BY-NC kidney and a CC BY-NC-SA inner ear under a CC BY-SA umbrella it cannot grant. |
| BodyParts3D | The website says CC BY 4.0; the README inside the data distribution says CC BY-SA 2.1 Japan. Unresolved. |
| TurboSquid, CGTrader, Fab, Sketchfab | Their licences forbid serving models in open WebGL formats by name, at every paid tier. |
| BioRender | Forbids use of its content outside the BioRender application. |
| Patrick J. Lynch skeleton | CC BY 2.5 is acceptable, but a public-domain alternative of better proportion existed, so no attribution obligation was taken on. |
