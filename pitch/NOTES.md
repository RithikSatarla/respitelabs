# Deck notes: type, palette, and what we borrowed

## Type pairing

**Georgia for headlines. Arial for everything else.**

A serif headline over a neutral sans body is a real editorial pairing, and it is
the fastest way to not look like a template. Decks built by a tool default to
sans on sans, usually Inter, so a serif headline reads as a decision someone
made. Georgia was drawn for screens, so it holds up at 40pt on a laptop and on a
projector.

Both fonts ship with Windows, macOS, PowerPoint and Google Slides. Nothing has to
be embedded and nothing substitutes to something ugly when Carlos opens it. That
ruled out the site's own fonts, Archivo and IBM Plex Mono, which are Google
Fonts and would fall back unpredictably in PowerPoint.

Numbers inside charts are Arial. Note that Georgia sets old-style figures, the
kind with descenders, so a number in a headline and the same number in a chart
do not look identical. We kept it: the headline reads as prose, which is what a
sentence-shaped headline wants, and the chart reads as data.

## Palette

Taken from the product's own site (`docs/index.html`), so the deck and the thing
it is selling look related.

| Use | Hex |
|---|---|
| Ink, all headlines and body | `#0C0C0E` |
| Secondary text, chart axes | `#505157` |
| Hairlines and rules | `#E3E3E6` |
| Panel fill | `#F6F6F7` |
| Accent, Kintrace only | `#E8662F` |
| GO green, the certify stamp only | `#0B7A2A` |

One accent. It marks Kintrace and nothing else. Anything that is not us, a
dashboard, a competitor, the old way, is grey. That means the eye finds us on
every chart without a legend doing the work.

No gradients. No second accent. White background everywhere.

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
