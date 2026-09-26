"""Pull the speaker notes out of the built deck into pitch/SPEAKER_NOTES.md.

The notes live in the pptx, written by pitch/build_deck.py. This keeps a
readable copy you can hold on a call without opening PowerPoint.

Run from the repo root:  python pitch/export_notes.py
"""

import pathlib

from pptx import Presentation

ROOT = pathlib.Path(__file__).resolve().parent.parent
DECK = ROOT / "pitch" / "Kintrace-deck.pptx"
OUT = ROOT / "pitch" / "SPEAKER_NOTES.md"


def headline_of(slide):
    """The first non-empty text on the slide, which is always the headline."""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            return " ".join(shape.text_frame.text.split())
    return "(no text)"


def main():
    prs = Presentation(str(DECK))
    out = [
        "# Speaker notes",
        "",
        "What to say on each slide, in the words you would actually use. These are",
        "also inside the pptx, under each slide, so they show up in presenter view.",
        "",
        "Generated from the deck by `python pitch/export_notes.py`. Edit the notes in",
        "`pitch/build_deck.py` and rebuild, or edit them in PowerPoint and they stay.",
        "",
    ]
    for i, slide in enumerate(prs.slides, start=1):
        head = headline_of(slide)
        notes = ""
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
        out.append(f"## Slide {i}. {head}")
        out.append("")
        out.append(notes if notes else "_No notes._")
        out.append("")
    OUT.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
