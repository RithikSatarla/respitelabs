# Deck notes: type, palette, and what we borrowed

## Type pairing

**Arial in the pptx. Archivo on the web version.**

The website's display face is Archivo, a grotesque, with IBM Plex Mono for
numbers. The web deck at `docs/pitch/` loads both from Google Fonts, so it
matches the site exactly.

The pptx cannot. Archivo is a Google Font, and embedding fonts in a pptx
substitutes unpredictably between PowerPoint and Google Slides, which is
exactly where Carlos will open it. So the pptx uses Arial, which is the
fallback the site's own CSS declares for Archivo. Same category of face, no
substitution risk, opens identically everywhere.

Numbers in the charts are Arial to match the pptx.

## Palette

The website's colours, so the deck and the product look related.

| Use | Hex |
|---|---|
| Paper, every slide background | `#F7F7F4` |
| Ink, all headlines and body | `#121212` |
| Secondary text, chart axes | `#55565B` |
| Hairlines and rules | `#DEDED8` |
| Panel fill | `#EFEFEA` |
| Safety yellow, Kintrace only | `#FFC400` |
| GO green, the certify stamp only | `#0B7A2A` |

**Yellow is a fill, never text and never a thin line.** `#FFC400` on `#F7F7F4`
has nowhere near the contrast to read as type. So it shows up as a block behind
black text, a bar, a rule or a filled shape. Where the old palette used orange
as a text colour, that text is now black and the yellow moved behind it as a
highlight. On the results chart the Kintrace bars are yellow with a hairline
black edge, because a yellow shape on off-white needs an edge to hold its form.
On the health chart the line is black and the yellow is the fill underneath it.

Anything that is not us is grey. That means the eye finds Kintrace on every
chart without a legend doing the work.

No gradients. No second accent.

## What we borrowed from decks that worked

Looked at write-ups of the Airbnb seed deck and Y Combinator's own deck advice.

**Airbnb, 2008, raised $600K.** What people keep pointing at is not the design.
It is that the problem slide was three short statements you cannot argue with,
and the opening line, "Book rooms with locals, rather than hotels", told you the
whole company in one sentence. We copied that shape: the cover says what happens
and what we do about it, in one line, and the problem is a story, not a list.
- https://www.failory.com/pitch-deck/airbnb
- https://mypitchdecks.com/case-studies/airbnb-pitch-deck

**Airbnb showed the model working in one provable place** before claiming a
market, using the Denver convention when hotels sold out. We do not have
customers, so our version of that is the one incident, with the real record the
tool printed. It is honest about being simulated and still concrete.

**Y Combinator's design advice: legibility, simplicity, obviousness.** One idea
per slide. Big type. Key text at the top. And the line we took most literally:
do not show a chart and let the investor work out the point, write the takeaway
as the headline.
- https://www.ycombinator.com/library/4T-how-to-design-a-better-pitch-deck

**So every headline on this deck is a full sentence that states the conclusion.**
"Kintrace found the cause every time, the dashboard found it 69% of the time" is
the headline. The chart underneath is evidence for a claim already made, which
means a reader who only reads headlines still gets the whole argument.

**Layout varies by what the slide has to do**, which is the other thing a
templated deck never does. A timeline slide, a full-bleed screenshot, a big
chart, a 2x2 and a stamp are all different shapes. The one thing that does not
move is the headline, same place, same size, on all eleven slides.

## Things we are deliberately not doing

- No "The Problem" / "The Solution" headers.
- No three stat tiles in a row. Every number sits in a sentence or a picture.
- No icons, no emoji, no numbered section labels.
- No claim of a customer, a pilot, a partnership or a real-robot result, because
  we have none. Every measured number on the deck says simulation.
