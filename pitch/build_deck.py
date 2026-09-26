"""Build pitch/Kintrace-deck.pptx.

Type, palette and the reasoning behind them are in pitch/NOTES.md.
Every outside number is in pitch/SOURCES.md. Speaker notes are written into the
pptx and also exported to pitch/SPEAKER_NOTES.md by pitch/export_notes.py.

Run from the repo root:  python pitch/build_deck.py
"""

import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHARTS = ROOT / "pitch" / "charts"
OUT = ROOT / "pitch" / "Kintrace-deck.pptx"

# --- design system -------------------------------------------------------
HEAD_FONT = "Georgia"
BODY_FONT = "Arial"

INK = RGBColor(0x0C, 0x0C, 0x0E)
SOFT = RGBColor(0x50, 0x51, 0x57)
FAINT = RGBColor(0x76, 0x77, 0x7D)
RULE = RGBColor(0xE3, 0xE3, 0xE6)
PANEL = RGBColor(0xF6, 0xF6, 0xF7)
ACCENT = RGBColor(0xE8, 0x66, 0x2F)
GO = RGBColor(0x0B, 0x7A, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = 13.333, 7.5
ML, MR = 0.85, 0.85
CW = SW - ML - MR
HEAD_TOP = 0.56
BODY_TOP = 1.86          # everything below the headline starts here
SRC_TOP = 6.92

NOTES = {}


def deck():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    return prs


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def write(tf, lines, font=BODY_FONT, size=18, color=INK, bold=False,
          spacing=1.25, space_after=0, align=PP_ALIGN.LEFT):
    """lines: a string, or a list of strings / (text, overrides) tuples."""
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        over = {}
        if isinstance(item, tuple):
            item, over = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = over.get("align", align)
        p.line_spacing = over.get("spacing", spacing)
        if space_after or over.get("space_after"):
            p.space_after = Pt(over.get("space_after", space_after))
        r = p.add_run()
        r.text = item
        f = r.font
        f.name = over.get("font", font)
        f.size = Pt(over.get("size", size))
        f.color.rgb = over.get("color", color)
        f.bold = over.get("bold", bold)
    return tf


def headline(slide, text, size=29):
    tf = box(slide, ML, HEAD_TOP, CW, 1.05)
    write(tf, text, font=HEAD_FONT, size=size, color=INK, spacing=1.16)
    return tf


def source(slide, text):
    tf = box(slide, ML, SRC_TOP, CW, 0.4)
    write(tf, text, size=9.5, color=FAINT, spacing=1.2)


def rule(slide, x, y, w, color=RULE, thick=0.012):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(thick))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def panel(slide, x, y, w, h, fill=PANEL):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def dot(slide, x, y, d, fill=ACCENT):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y),
                               Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(CHARTS / name), Inches(x), Inches(y), **kw)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


# --- slides --------------------------------------------------------------

def s01_cover(prs):
    s = blank(prs)
    rule(s, ML, 1.55, 1.5, ACCENT, thick=0.055)

    tf = box(s, ML, 1.95, CW, 1.5)
    write(tf, "Kintrace", font=HEAD_FONT, size=76, color=INK, spacing=1.0)

    tf = box(s, ML, 3.35, 9.3, 1.7)
    write(tf, "Robots get knocked out of place and nobody notices. "
              "Kintrace finds what moved and puts it back.",
          font=HEAD_FONT, size=23, color=SOFT, spacing=1.34)

    rule(s, ML, 5.85, CW)
    tf = box(s, ML, 6.08, 7.0, 0.9)
    write(tf, [("Rithik Satarla", {"bold": True}),
               "rithiksatarla@gmail.com"],
          size=14, color=INK, spacing=1.38)

    tf = box(s, ML + 7.4, 6.08, CW - 7.4, 0.9)
    write(tf, "Previously LoopCell.\nSame product, new name.",
          size=12, color=FAINT, spacing=1.38, align=PP_ALIGN.RIGHT)

    notes(s, """
Hi, I am Rithik. Kintrace is physical observability for robot arms.

Here is the one line. A robot is a machine that believes things about itself.
It believes where its camera is, how long its gripper is, where its joints are
zeroed. When one of those beliefs stops being true, the robot does not know.
It keeps working, and it keeps getting it wrong.

Kintrace is the thing that checks those beliefs against reality, and when one
is wrong, tells you which one, by how much, and writes the correction back.

If you saw this a few months ago it was called LoopCell. Same product.
""")


def s02_moment(prs):
    s = blank(prs)
    headline(s, "At 1:40 a cart clips a camera. Nobody finds out until 2:10.")

    rows = [
        ("Today", SOFT, RULE, [
            ("1:40", "A cart clips the camera mount. Nothing alarms."),
            ("1:40 on", "The robot keeps working. Every grab is now a centimetre off."),
            ("2:10", "Grabs start missing. The line stops."),
            ("Hours later", "A technician arrives and checks each possible cause by hand."),
        ]),
        ("With Kintrace", ACCENT, ACCENT, [
            ("1:40:23", "Flagged. The camera moved 18 mm."),
            ("1:40:29", "A 6 second check motion confirms nothing else moved."),
            ("1:41", "The new camera calibration is written to the controller."),
            ("1:43", "Six test grabs, all good. Signed off, back in production."),
        ]),
    ]

    y = BODY_TOP
    for title, colour, dotcol, items in rows:
        tf = box(s, ML, y, 2.2, 0.4)
        write(tf, title, size=15, color=colour, bold=True)

        line_y = y + 0.66
        rule(s, ML + 0.06, line_y, CW - 0.5, RULE, thick=0.01)

        step = (CW - 0.7) / 4
        for i, (when, what) in enumerate(items):
            cx = ML + 0.02 + i * step
            dot(s, cx, line_y - 0.055, 0.13, dotcol)
            tf = box(s, cx - 0.02, line_y + 0.24, step - 0.26, 1.35)
            write(tf, [(when, {"bold": True, "size": 13.5, "color": INK}),
                       (what, {"size": 13, "color": SOFT, "spacing": 1.3})],
                  spacing=1.3)
        y += 2.42

    tf = box(s, ML, SRC_TOP, CW, 0.4)
    write(tf, "An illustration. The 18 mm and the timings are printed by the tool, "
              "running on our simulator. No cell like this exists yet.",
          size=11.5, color=FAINT)

    notes(s, """
This is the whole company in one slide, so let me walk it.

Top row is what happens today. A cart clips the camera mount at 1:40. Nothing
alarms, because from the software's point of view nothing failed. The robot is
still running, still reporting healthy. But every grab is now about a
centimetre off. If this is a factory, parts start getting missed at 2:10 and
the line stops. If this is a company teaching robots, every recording made in
that half hour is quietly wrong, and it goes into the training set.

Then a technician drives out and checks each possible cause by hand. Camera?
Tool? Joints? Timing? That is hours.

Bottom row is with Kintrace. Flagged at 1:40:23, twenty three seconds in, off
the robot's own logs. It runs a six second check motion to confirm the camera
moved and nothing else did. It writes the new calibration. Six test grabs.
Signed off.

Be straight about this: this is an example from our simulator. The times and
the 18 mm are real output from the tool, not numbers I made up for the slide.
""")


def s03_cost(prs):
    s = blank(prs)
    headline(s, ["You pay for it twice: in ruined training data,",
                 "and in stopped lines."])

    picture(s, "cost_episodes.png", ML, BODY_TOP + 0.12, w=6.55)
    tf = box(s, ML, BODY_TOP + 2.28, 6.3, 1.4)
    write(tf, [("Robot training data", {"bold": True, "size": 16}),
               ("An open audit of 4,959 recorded episodes flagged 15.6% of them "
                "for exclusion, and another half for review.",
                {"size": 15, "color": SOFT})],
          spacing=1.34)

    rule(s, 8.05, BODY_TOP + 0.2, 0.012, RULE)
    vline = slide_vline(s, 8.05, BODY_TOP + 0.15, 2.95)

    tf = box(s, 8.5, BODY_TOP + 0.18, 4.0, 0.88)
    write(tf, "$2.3M", font=HEAD_FONT, size=54, color=ACCENT, spacing=1.0)
    tf = box(s, 8.5, BODY_TOP + 1.14, 4.0, 2.1)
    write(tf, [("an hour", {"bold": True, "size": 17}),
               ("Up to, for an idle production line at a big car plant. Across "
                "the 500 largest manufacturers, unplanned downtime of every kind "
                "costs $1.4 trillion a year, about 11% of revenue.",
                {"size": 15, "color": SOFT})],
          spacing=1.34)

    source(s, "Episode audit: an open source tool run over 12 public datasets, dev.to, 2026. "
              "Downtime: Siemens, The True Cost of Downtime 2024. Both linked in pitch/SOURCES.md.")

    notes(s, """
Two different buyers feel this, and they feel it differently.

On the left, companies teaching robots. When a robot is physically out of
place, it still records. Those recordings look fine. They are labelled with
where the robot thought it was, not where it actually was. Someone audited
nearly five thousand public robot training episodes with an open tool and
marked about one in six as ones you should throw out. I want to be careful
here: that tool flags episodes for review, it is not proof, and the author
says so. But it points at something real. At a hundred and eighteen dollars an
hour to collect this data, throwing out one in six is expensive.

On the right, factories. An idle line at a big car plant runs to two point
three million dollars an hour. The five hundred biggest manufacturers lose
about one point four trillion a year to unplanned downtime.

I am not claiming Kintrace recovers all of that. I am saying the minutes
between a bump and someone noticing are worth a lot, and right now nobody is
selling those minutes back.
""")


def slide_vline(slide, x, y, h):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                               Inches(0.012), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = RULE
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def s04_why_unsolved(prs):
    s = blank(prs)
    headline(s, ["Everything on the market tells you something is wrong,",
                 "not what moved."])

    cols = [
        ("Monitoring dashboards",
         "Ember, Foxglove, Formant",
         "They watch the software and the logs. They will tell you a cell is "
         "slow or a run failed. They do not measure where the camera is, so "
         "they cannot name the physical cause."),
        ("Calibration hardware",
         "Dynalog, CAPTRON, Renishaw",
         "These do measure, and they are accurate. But you bolt a device into "
         "every cell you want covered, and each one mostly does the tool."),
        ("A technician",
         "How it is actually done",
         "Someone drives out and rules out each cause by hand, one at a time, "
         "with the cell down while they do it. It works. It costs hours."),
    ]

    w = (CW - 1.5) / 3
    for i, (title, who, body) in enumerate(cols):
        x = ML + i * (w + 0.75)
        rule(s, x, BODY_TOP, 1.05, ACCENT if i == 2 else RULE, thick=0.035)
        tf = box(s, x, BODY_TOP + 0.34, w, 0.5)
        write(tf, title, size=19, color=INK, bold=True, font=HEAD_FONT)
        tf = box(s, x, BODY_TOP + 0.95, w, 0.35)
        write(tf, who, size=12.5, color=FAINT)
        tf = box(s, x, BODY_TOP + 1.42, w, 2.6)
        write(tf, body, size=15, color=SOFT, spacing=1.42)

    panel(s, ML, 5.78, CW, 0.92)
    tf = box(s, ML + 0.38, 5.99, CW - 0.76, 0.6)
    write(tf, "We have not found anyone doing all of it from software alone: when it "
              "changed, what moved, by how much, and the fix written back.",
          size=16, color=INK, spacing=1.3)

    source(s, "What each of these does is taken from their own sites and materials. "
              "Listed in pitch/SOURCES.md.")

    notes(s, """
Why has nobody done this? They have each done a piece.

Monitoring dashboards. Ember is the closest in spirit, YC backed, and their
line is literally "know what changed, know what failed". Foxglove raised forty
million last November. These are good products. But they watch the software.
They can tell you a run failed. They cannot tell you the camera is eleven
millimetres to the left, because they never measure where the camera is.

Calibration hardware. Dynalog sells exactly this outcome, restore accuracy
after a crash without re-teaching. It works. The catch is it is hardware you
mount in the cell, and it is mostly about the tool tip. A fleet with two
hundred cells is not buying two hundred fixtures.

And then there is what actually happens most of the time, which is a person
driving out and checking things by hand.

So the gap is the whole sentence: when it changed, what moved, by how much,
and the fix, from software, with no new hardware.
""")


def s05_product(prs):
    s = blank(prs)
    headline(s, ["Kintrace reads the robot's own logs, then says",
                 "what moved and writes the fix."])

    pic = picture(s, "play_record.png", ML, BODY_TOP + 0.05, w=8.6)

    steps = [
        ("Notices", "It finds the moment things changed and lines it up with "
                    "the protective stop."),
        ("Explains", "It names the cause and sizes it. Here: the camera, 18.1 mm."),
        ("Fixes", "It writes the new calibration, re-checks, runs six test "
                  "grabs, then signs GO or NO-GO."),
    ]
    x = ML + 8.95
    y = BODY_TOP + 0.18
    for i, (title, body) in enumerate(steps):
        dot(s, x, y + 0.08, 0.15, ACCENT)
        tf = box(s, x + 0.34, y - 0.02, 2.85, 0.4)
        write(tf, title, size=17, color=INK, bold=True, font=HEAD_FONT)
        tf = box(s, x + 0.34, y + 0.4, 2.85, 1.05)
        write(tf, body, size=13.5, color=SOFT, spacing=1.36)
        y += 1.62

    tf = box(s, ML, SRC_TOP, 8.6, 0.4)
    write(tf, "A real run of the tool, on our simulated cell. Try it yourself at "
              "the link on the last slide.", size=11.5, color=FAINT)

    notes(s, """
This is the actual thing, not a mockup. This is the record Kintrace printed
for the incident on slide two.

Three things happen. It notices: it found the change at 13:40:23 and matched it
to the protective stop the controller already logged, so you are not guessing
which event mattered.

It explains: camera moved 18.1 millimetres. Not "camera fault", not "anomaly
detected". A distance. And it says the other four things it checked did not
move, which is the hard part. A bumped camera and a drifted joint look almost
identical in the logs. The difference is that a moved camera shifts everything
by the same amount, while a joint that is off gives you an error that changes
with the arm's pose. Kintrace fits both explanations and keeps the one that
brings the error down to sensor noise.

Then it fixes: writes the calibration, runs the check again, does six test
grabs, and signs GO.

There is no model in here. It is geometry and statistics. Same log in, same
answer out, every time, and you can check the arithmetic. For a buyer who has
to certify a cell, that matters more than accuracy.
""")


def s06_proof(prs):
    s = blank(prs)
    headline(s, ["In our simulator, Kintrace named the right cause",
                 "every time. The dashboard baseline we wrote got 69%."])

    picture(s, "results_by_fault.png", 1.97, BODY_TOP + 0.08, h=4.70)

    source(s, "120 runs on our simulated UR5e pick cell, 20 per case, every fault sized to cause the same "
              "20 mm miss. The baseline is ours, built from the signals a typical dashboard tracks. Never run on a real arm.")

    notes(s, """
This is our benchmark. A hundred and twenty runs, twenty for each of the five
faults plus twenty healthy runs. Every fault is sized so it causes the same
twenty millimetre miss, so the robot looks equally broken in all of them and
the only question is whether you can tell them apart.

Orange is Kintrace, grey is the baseline. Look at the top two rows. A joint
knocked off zero: the dashboard gets it right one time in five. A wrong tool
setting loaded: two in five. Those are the expensive ones to diagnose by hand,
and they are exactly where the dashboard is worst, because from its point of
view all of these look like the same thing, which is that grabs are missing.

Two honest caveats, and I would rather say them than have you find them.
One, this is our simulator. Two, we wrote the baseline as well, so it is not a
benchmark against a shipping product, it is a benchmark against the signals a
typical dashboard tracks.

That is why the money in this round goes to running exactly this on a real arm.
""")


def s07_early(prs):
    s = blank(prs)
    headline(s, ["It also catches slow drift, about 55 seconds",
                 "before the first grab misses."])

    picture(s, "early_warning.png", 1.80, BODY_TOP + 0.05, h=4.70)

    source(s, "One simulated run of a sagging camera mount. Across 20 drifts it warned before the first "
              "miss 19 times, named the cause right 20 times, and raised 0 false alarms in 230 healthy "
              "windows. Drift is sped up so a run takes minutes. We have not measured lead time at "
              "real-world drift rates.")

    notes(s, """
The last slide was after a crash. This one is the other half, and honestly it
is the half that makes it a subscription rather than a callout fee.

Kintrace reads the logs the robot already writes, with no check motion and no
interruption, and scores the cell every ten seconds. This run is a camera mount
slowly sagging. The score falls. At sixty seconds Kintrace says: your camera is
moving, parts are two and a half millimetres off, at this rate you start
missing in about a minute. At that point nothing has failed yet. The first
missed grab is fifty five seconds later.

Across twenty of these drifts it warned in time nineteen times, named the cause
right all twenty, and gave zero false alarms across two hundred and thirty
healthy windows. The zero matters most. A monitoring tool that cries wolf gets
switched off in a week.

The drift here is sped up so a run takes minutes instead of weeks. Real drift
is slower, which means more warning, not less.
""")


def s08_why_now(prs):
    s = blank(prs)
    headline(s, ["There have never been more robots, and the money",
                 "behind them doubled in three years."])

    picture(s, "why_now.png", ML, BODY_TOP + 0.1, w=7.9)

    x = ML + 8.35
    facts = [
        ("5 million", "industrial robots now operating, a record, up 9% in a "
                      "year. The US installed 38,500 and passed Japan. "
                      "IFR, September 2026."),
        ("47% a year", "forecast growth for physical AI, from $0.89B in 2025 to "
                       "$15.2B by 2032. Every robot in it learns from recordings."),
    ]
    y = BODY_TOP + 0.16
    for value, body in facts:
        rule(s, x, y, 0.95, ACCENT, thick=0.035)
        tf = box(s, x, y + 0.26, 3.6, 0.68)
        write(tf, value, font=HEAD_FONT, size=30, color=INK, spacing=1.05)
        tf = box(s, x, y + 0.95, 3.6, 1.28)
        write(tf, body, size=14, color=SOFT, spacing=1.4)
        y += 2.42

    source(s, "IFR World Robotics 2026. MarketsandMarkets, physical AI. "
              "Venture figures from Robotics Center, State of Robotics 2026. Links in pitch/SOURCES.md.")

    notes(s, """
Why this is the right moment, and not two years ago.

Three things landed at once. First, two days ago the IFR published its 2026
report: five million industrial robots now operating, a record, up nine
percent. The US installed thirty eight and a half thousand and passed Japan to
become the second largest market. The installed base I sell into is at its
largest ever and still growing.

Second, the chart. Venture money into robotics went from three point eight
billion in 2022 to nine point four billion in 2025, up forty one percent in the
last year alone. That money buys arms, and every arm it buys is a cell someone
now has to keep straight.

Third, and this is the one I care about most, the physical AI market is
forecast to grow about forty seven percent a year. That whole category is
companies teaching robots from recordings. They are the buyer who feels a
misaligned camera immediately, because it silently poisons the thing they are
spending the money to build.

Two years ago I would have been selling to factories only. Now there is a
second buyer who is faster to move and in more of a hurry.
""")


def s09_market(prs):
    s = blank(prs)
    headline(s, ["At $200 a robot a month, the part we can",
                 "realistically sell to is $1.2B a year."])

    picture(s, "market.png", 1.92, BODY_TOP + 0.02, h=3.92)   # 2.42 aspect -> 9.49in wide, centred

    panel(s, ML, 5.98, CW, 0.82)
    tf = box(s, ML + 0.38, 6.15, CW - 0.76, 0.5)
    write(tf, [("Top down check:  ", {"bold": True, "size": 15}),
               ("$22.5B was spent on industrial robot services in 2023, heading "
                "to $41.6B by 2033. Our $12B bottom up number sits inside a "
                "services market that is already twice its size.",
                {"size": 15, "color": SOFT})],
          spacing=1.3)
    merge_runs_inline(tf)

    source(s, "Robot count: IFR World Robotics 2026. Services market: Market.us, October 2024. "
              "The 10% camera share is our own estimate; we looked for a public figure and did not find one.")

    notes(s, """
Bottom up first, because I would rather show the arithmetic than a big number.

Two hundred dollars a robot a month is twenty four hundred a year. Five million
robots is twelve billion a year if every robot on earth bought it. That is the
TAM and it is not a real number, nobody sells to everybody.

The middle circle is the one I actually believe. Kintrace needs the robot to
have a camera looking at the work area. My estimate is about one in ten, which
gets me to one point two billion a year. I want to flag that ten percent is my
estimate. I looked for a public figure for what share of the installed robot
base is vision guided and there is not one. The public numbers size the robot
vision market in dollars, which is a different question. So treat that as the
number to push me on.

The small dot is twenty thousand robots five years out, forty eight million a
year. That is the number I am actually trying to hit.

As a check from the other direction: twenty two and a half billion a year is
already spent on industrial robot services, maintenance and optimisation,
heading to forty one billion. My twelve billion bottom up sits comfortably
inside a market that already exists and already pays for this outcome.
""")


def merge_runs_inline(tf):
    """Put the two runs of the first paragraph on one line."""
    p = tf.paragraphs[0]
    if len(tf.paragraphs) > 1:
        extra = tf.paragraphs[1]
        for r in list(extra.runs):
            p._p.append(r._r)
        extra._p.getparent().remove(extra._p)


def s10_business(prs):
    s = blank(prs)
    headline(s, ["$200 a robot a month, against a technician visit",
                 "that costs $1,500 to $3,000."])

    # Left: pricing and go to market.
    tf = box(s, ML, BODY_TOP, 4.55, 0.4)
    write(tf, "How we charge", size=13, color=FAINT, bold=True)
    items = [
        ("$200", "per robot, per month, for always-on watching"),
        ("~$300", "per incident we find and fix. Planned pilot pricing, no pilots yet"),
        ("$1,500 to $3,000", "what the technician visit it replaces costs today (our estimate)"),
    ]
    y = BODY_TOP + 0.44
    for value, body in items:
        tf = box(s, ML, y, 4.5, 0.38)
        write(tf, value, font=HEAD_FONT, size=21, color=ACCENT if value != "$1,500 to $3,000" else SOFT)
        tf = box(s, ML, y + 0.38, 4.05, 0.6)
        write(tf, body, size=13, color=SOFT, spacing=1.3)
        y += 0.98

    tf = box(s, ML, y + 0.12, 4.55, 0.4)
    write(tf, "Who we sell to, in order", size=13, color=FAINT, bold=True)
    tf = box(s, ML, y + 0.54, 4.05, 0.95)
    write(tf, "Robot learning teams first, they feel bad data fastest. "
              "Then robots-as-a-service fleets, who pay for every callout. "
              "Then robot makers.", size=13.5, color=SOFT, spacing=1.38)

    # Right: the 2x2.
    gx, gy, gw, gh = 6.5, BODY_TOP + 0.35, 5.5, 3.85
    panel(s, gx, gy, gw, gh, RGBColor(0xFA, 0xFA, 0xFB))
    rule(s, gx, gy + gh / 2, gw, RULE, thick=0.012)
    vl = slide_vline(s, gx + gw / 2, gy, gh)

    def cell(cx, cy, w, name, line, accent=False):
        tf = box(s, cx, cy, w, 0.34)
        write(tf, name, size=13.5, color=ACCENT if accent else INK, bold=True)
        tf = box(s, cx, cy + 0.32, w, 0.7)
        write(tf, line, size=11, color=SOFT, spacing=1.28)

    cell(gx + 0.28, gy + 0.34, 2.25, "Dynalog, CAPTRON",
         "Finds and fixes it, but you mount hardware in every cell.")
    cell(gx + gw / 2 + 0.28, gy + 0.34, 2.3, "Kintrace",
         "Finds it, sizes it, writes the fix, certifies the cell. Software only.",
         accent=True)
    cell(gx + 0.28, gy + gh / 2 + 0.34, 2.25, "Robot makers",
         "Built-in mastering routines. One brand, run by a technician.")
    cell(gx + gw / 2 + 0.28, gy + gh / 2 + 0.34, 2.3, "Ember, Foxglove, dataset checkers",
         "Show you something is off. No geometry, no fix.")

    tf = box(s, gx, gy - 0.34, gw, 0.3)
    write(tf, "Finds and fixes the cause", size=11.5, color=FAINT, bold=True,
          align=PP_ALIGN.CENTER)
    tf = box(s, gx, gy + gh + 0.06, gw, 0.3)
    write(tf, "Only shows a problem", size=11.5, color=FAINT, bold=True,
          align=PP_ALIGN.CENTER)
    tf = box(s, gx - 1.24, gy + gh / 2 - 0.18, 1.16, 0.3)
    write(tf, "Needs hardware", size=11.5, color=FAINT, bold=True, align=PP_ALIGN.RIGHT)
    tf = box(s, gx + gw + 0.06, gy + gh / 2 - 0.18, 1.2, 0.3)
    write(tf, "Software only", size=11.5, color=FAINT, bold=True)

    source(s, "The $1,500 to $3,000 technician visit is our own estimate from conversations, "
              "not a published figure. Our pricing is intended, not yet tested on a buyer.")

    notes(s, """
Pricing. Two hundred a robot a month for the always-on watching. In pilots I
will also do it per incident, about three hundred dollars for a fix, because
that is an easier first yes and it prices against the thing it replaces.

The thing it replaces is a technician visit, which I put at fifteen hundred to
three thousand dollars. I want to be clear that is my estimate from
conversations, not a published figure. If you have a better number I would
genuinely like it.

Order of customers. Robot learning teams first. They feel bad data fastest,
they have budget right now, and they buy software without a procurement cycle.
Then robots-as-a-service fleets, who eat the cost of every callout themselves,
so the maths is immediate. Robot makers last, because they move slowly, but
they are where this ends up if it works.

On the right is the landscape. Horizontal is whether you need hardware in the
cell. Vertical is whether you find and fix the cause or just flag it. Dynalog
is top left: they genuinely fix it, but it is a device per cell. Ember and
Foxglove are bottom right: software, but they tell you something is off. The
top right box, software only and actually names and fixes the cause, is empty.
That is the whole bet.
""")


def s11_ask(prs):
    s = blank(prs)
    headline(s, ["Raising [$150K] to put Kintrace on real arms",
                 "and into three paid pilots."])

    tf = box(s, ML, BODY_TOP, 5.1, 0.4)
    write(tf, "Where it goes", size=13, color=FAINT, bold=True)
    tf = box(s, ML, BODY_TOP + 0.45, 5.0, 2.4)
    write(tf, ["A desk arm rig, then time on a lab arm, to reproduce all five "
               "faults on real hardware.",
               "Three paid pilots with robot learning teams.",
               "A reader so fleets can send their own robot logs, in the formats "
               "they already record."],
          size=14.5, color=SOFT, spacing=1.42, space_after=11)

    tf = box(s, ML, BODY_TOP + 3.02, 5.1, 0.4)
    write(tf, "Team", size=13, color=FAINT, bold=True)
    tf = box(s, ML, BODY_TOP + 3.45, 5.0, 1.3)
    write(tf, [("Rithik Satarla", {"bold": True, "size": 15, "color": INK}),
               ("Physical AI research intern at CMU. Harvard Innovation Labs. "
                "Wrote all of the above.", {"size": 13.5, "color": SOFT}),
               ("[Cofounder]", {"size": 13.5, "color": FAINT})],
          spacing=1.38)

    # Milestones down the right.
    tf = box(s, 6.9, BODY_TOP, 5.6, 0.4)
    write(tf, "Next six months", size=13, color=FAINT, bold=True)

    miles = [
        ("Nov 2", "YC W27 application in", True),
        ("Month 1-2", "Five faults reproduced on the desk arm, numbers published", False),
        ("Month 3-4", "Same five on a lab arm, and the first paid pilot", False),
        ("Month 5-6", "Three paid pilots running, real log reader shipped", False),
    ]
    y = BODY_TOP + 0.52
    for when, what, hot in miles:
        dot(s, 6.92, y + 0.07, 0.14, ACCENT if hot else RULE)
        tf = box(s, 7.34, y - 0.04, 5.1, 0.34)
        write(tf, when, size=13.5, color=INK, bold=True)
        tf = box(s, 7.34, y + 0.3, 5.1, 0.6)
        write(tf, what, size=13.5, color=SOFT, spacing=1.32)
        y += 1.06

    rule(s, ML, 6.42, CW)
    tf = box(s, ML, 6.62, CW, 0.5)
    write(tf, "rithiksatarla@gmail.com    ·    Try it: [live demo link]    ·    "
              "Code: github.com/RithikSatarla/kintrace",
          size=12.5, color=SOFT)

    notes(s, """
I am raising a hundred and fifty thousand. Here is exactly what it buys.

First, real hardware. Everything I showed you is simulation and I have said so
on every slide. The single biggest risk in this company is that real sensors
behave worse than my simulator. So the first money goes to a desk arm rig,
about three hundred dollars of parts, where I cause all five faults by hand and
publish what happens, including whatever breaks. Then time on a proper lab arm.

Second, three paid pilots with robot learning teams. Paid, because I want to
know if this is a vitamin or a painkiller, and the only honest test is whether
someone hands over money.

Third, a reader for real robot logs, so a fleet can send me what they already
record instead of adapting to my format.

Timeline on the right. The YC W27 application goes in on the second of
November. Five faults on the desk arm in the first two months, a lab arm and
the first paid pilot by month four, three pilots running by month six.

On the team: it is me right now. I am a physical AI research intern at CMU and
I am at the Harvard Innovation Labs, and I wrote all of this. I am looking for
a cofounder and I would rather say that plainly than pretend otherwise.

You can run the thing yourself at the demo link before you decide.
""")


def main():
    prs = deck()
    for fn in (s01_cover, s02_moment, s03_cost, s04_why_unsolved, s05_product,
               s06_proof, s07_early, s08_why_now, s09_market, s10_business, s11_ask):
        fn(prs)
    prs.save(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
