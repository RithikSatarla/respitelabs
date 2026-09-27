"""Build pitch/Kintrace-deck.pptx.

Type, palette and the reasoning behind them are in pitch/NOTES.md.
Every outside number is in pitch/SOURCES.md. Speaker notes are written into the
pptx and exported to pitch/SPEAKER_NOTES.md by pitch/export_notes.py.

Run from the repo root:  python pitch/build_deck.py
"""

import pathlib

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHARTS = ROOT / "pitch" / "charts"
OUT = ROOT / "pitch" / "Kintrace-deck.pptx"

# --- design system, matching the website -------------------------------
# The website's faces, so the deck matches it.
#
# Both are Google Fonts. PowerPoint will substitute them on a machine that does
# not have them installed, so the PDF, which is rendered from the web version
# with the real fonts embedded, is the safe file to send. The TTFs are in
# pitch/fonts/ if you want to install them locally.
FONT = "Schibsted Grotesk"
MONO = "JetBrains Mono"

PAPER = RGBColor(0xF7, 0xF7, 0xF4)
INK = RGBColor(0x12, 0x12, 0x12)
SOFT = RGBColor(0x55, 0x56, 0x5B)
FAINT = RGBColor(0x7A, 0x7B, 0x7F)
RULE = RGBColor(0xDE, 0xDE, 0xD8)
PANEL = RGBColor(0xEF, 0xEF, 0xEA)
ACCENT = RGBColor(0xFF, 0xC4, 0x00)   # fill only, never text
GO = RGBColor(0x0B, 0x7A, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = 13.333, 7.5
ML, MR = 0.85, 0.85
CW = SW - ML - MR
HEAD_TOP = 0.56
BODY_TOP = 1.92
SRC_TOP = 6.92


def deck():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)
    return prs


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = PAPER
    return s


def box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def write(tf, lines, size=17, color=INK, bold=False, spacing=1.3,
          space_after=0, align=PP_ALIGN.LEFT, font=FONT):
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        over = {}
        if isinstance(item, tuple):
            item, over = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = over.get("align", align)
        p.line_spacing = over.get("spacing", spacing)
        sa = over.get("space_after", space_after)
        if sa:
            p.space_after = Pt(sa)
        r = p.add_run()
        r.text = item
        f = r.font
        f.name = over.get("font", font)
        f.size = Pt(over.get("size", size))
        f.color.rgb = over.get("color", color)
        f.bold = over.get("bold", bold)
    return tf


def headline(slide, text, size=28):
    tf = box(slide, ML, HEAD_TOP, CW, 1.1)
    write(tf, text, size=size, color=INK, bold=True, spacing=1.18)
    return tf


def source(slide, text):
    write(box(slide, ML, SRC_TOP, CW, 0.4), text, size=9.5, color=FAINT,
          spacing=1.25)


def shape(slide, kind, x, y, w, h, fill=None, line=None, lw=0.8):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    return s


def rule(slide, x, y, w, color=RULE, thick=0.012):
    return shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, thick, fill=color)


def vrule(slide, x, y, h, color=RULE, thick=0.012):
    return shape(slide, MSO_SHAPE.RECTANGLE, x, y, thick, h, fill=color)


def panel(slide, x, y, w, h, fill=PANEL):
    return shape(slide, MSO_SHAPE.RECTANGLE, x, y, w, h, fill=fill)


def dot(slide, x, y, d, fill=ACCENT, line=INK):
    return shape(slide, MSO_SHAPE.OVAL, x, y, d, d, fill=fill, line=line, lw=0.8)


def picture(slide, name, x, y, w=None, h=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(CHARTS / name), Inches(x), Inches(y), **kw)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text.strip()


ONE_LINER = ("Kintrace finds what moved on a robot, fixes it, "
             "and tells you it is good to go.")


# ---------------------------------------------------------------- slides

def s01_cover(prs):
    s = blank(prs)
    shape(s, MSO_SHAPE.RECTANGLE, ML, 0.78, 1.55, 0.085, fill=ACCENT)

    write(box(s, ML, 1.08, 8.0, 1.1), "Kintrace", size=58, bold=True,
          spacing=1.0)
    write(box(s, ML, 2.24, 5.35, 1.4), ONE_LINER, size=19, color=SOFT,
          spacing=1.36)

    # The cell drawing, with the tool's own result printed under it.
    picture(s, "cell_crashed.png", 6.60, 1.96, w=5.88)

    bar_y = 5.44
    shape(s, MSO_SHAPE.RECTANGLE, 6.60, bar_y, 4.62, 0.46, fill=INK)
    write(box(s, 6.82, bar_y + 0.12, 4.3, 0.3),
          "camera moved 18.1 mm  ·  fixed", size=12.5, color=WHITE, bold=True,
          font=MONO)
    shape(s, MSO_SHAPE.RECTANGLE, 11.22, bar_y, 1.26, 0.46, fill=ACCENT)
    write(box(s, 11.22, bar_y + 0.10, 1.26, 0.3), "GO", size=15,
          bold=True, align=PP_ALIGN.CENTER)
    write(box(s, 6.60, bar_y + 0.62, 5.88, 0.3),
          "The tool's own output, from a run on our simulator.",
          size=10.5, color=FAINT)

    rule(s, ML, 5.34, 5.35)
    write(box(s, ML, 5.58, 5.35, 1.3),
          [("Rithik Satarla", {"bold": True, "size": 15, "space_after": 3}),
           ("rithiksatarla@gmail.com", {"size": 13, "color": SOFT,
                                        "space_after": 11}),
           ("Previously LoopCell. Same product, new name.",
            {"size": 11.5, "color": FAINT})],
          spacing=1.34)

    notes(s, """
Hi, I am Rithik. Kintrace is physical observability for robot arms.

One line: a robot is a machine that believes things about itself. Where its
camera is, how long its gripper is, where its joints are zeroed. When one of
those beliefs stops being true, the robot does not know. It keeps working and
it keeps getting it wrong.

Kintrace checks those beliefs against reality. When one is wrong it tells you
which one, by how much, and writes the correction back.

That picture is our simulated cell at the moment of a collision, and the bar
underneath is the actual output: camera moved 18.1 millimetres, fixed, good
to go.

If you saw this a few months ago it was called LoopCell. Same product.
""")


def s02_what_we_do(prs):
    s = blank(prs)
    headline(s, ["Kintrace finds what moved on a robot, fixes it,",
                 "and tells you it is good to go."])

    steps = [
        ("Notices",
         "It sees the exact moment a robot got knocked out of place, from "
         "data the robot already records.",
         "13:40:23", "something changed"),
        ("Explains",
         "It says what moved and by how much. Not an alert. A measurement.",
         "18.1 mm", "the camera moved"),
        ("Fixes",
         "It updates the robot, runs test grabs, and says whether the cell "
         "is good to go.",
         "6 / 6", "test grabs, then GO"),
    ]

    w = (CW - 1.1) / 3
    for i, (title, body, big, cap) in enumerate(steps):
        x = ML + i * (w + 0.55)

        panel(s, x, BODY_TOP, w, 1.40)
        write(box(s, x + 0.26, BODY_TOP + 0.28, w - 0.5, 0.5), big,
              size=24, bold=True, spacing=1.0, font=MONO)
        write(box(s, x + 0.26, BODY_TOP + 0.90, w - 0.5, 0.3), cap,
              size=11.5, color=SOFT)

        shape(s, MSO_SHAPE.RECTANGLE, x, BODY_TOP + 1.70, 0.42, 0.055,
              fill=ACCENT)
        write(box(s, x, BODY_TOP + 1.94, w - 0.3, 0.4), title,
              size=19, bold=True)
        write(box(s, x, BODY_TOP + 2.42, w - 0.3, 1.35), body,
              size=14.5, color=SOFT, spacing=1.44)

    panel(s, ML, 6.02, CW, 0.74)
    write(box(s, ML + 0.34, 6.22, CW - 0.68, 0.4),
          "Software only. No new hardware. It works on robot arms that "
          "already have a camera.", size=15.5, bold=True)

    notes(s, """
If you remember nothing else from this deck, remember this slide.

Three things happen, in order.

It notices. A robot already writes a lot of data: commanded joint positions,
measured joint positions, what the camera sees, whether each grab worked.
Kintrace reads that and finds the moment the physical setup changed. In our
example, 13:40:23, and it matches that to the protective stop the controller
already logged, so you are not guessing which event mattered.

It explains. This is the part nobody else does. It does not say "anomaly
detected". It says the camera moved 18.1 millimetres, and the tool and the
joints did not. A number you can act on.

It fixes. It writes the corrected calibration back to the controller, runs the
check again, does six test grabs, and signs the cell GO or NO-GO.

The bottom line matters for how fast we can sell. This is software. No device
to mount, no downtime to install it. If the arm has a camera pointed at the
work area, we can run.
""")


def s03_moment(prs):
    s = blank(prs)
    headline(s, "At 1:40 a cart clips a camera. Nobody finds out until 2:10.")

    cells = ["cell_healthy.png", "cell_crashed.png", "cell_fixed.png"]
    caps = [
        ("1:40", "Running fine",
         "The camera is where the calibration says it is."),
        ("1:40 to 2:10", "Bumped, and nobody knows",
         "Every grab is a centimetre off. The robot still reports healthy."),
        ("1:43 with Kintrace", "Found, fixed, signed off",
         "Camera moved 18.1 mm. New calibration written. Six test grabs."),
    ]

    w = (CW - 0.9) / 3
    h_img = w / 1.739
    for i, (img, (when, title, body)) in enumerate(zip(cells, caps)):
        x = ML + i * (w + 0.45)
        picture(s, img, x, BODY_TOP, w=w)

        y = BODY_TOP + h_img + 0.20
        shape(s, MSO_SHAPE.RECTANGLE, x, y, 0.42, 0.055,
              fill=ACCENT if i == 2 else RULE)
        write(box(s, x, y + 0.20, w, 0.3), when, size=12.5, color=SOFT,
              bold=True)
        write(box(s, x, y + 0.56, w, 0.34), title, size=16, bold=True)
        write(box(s, x, y + 0.96, w - 0.2, 0.85), body, size=13, color=SOFT,
              spacing=1.4)

    source(s, "An illustration, drawn from a run on our simulator. The 18.1 mm and the timings are printed "
              "by the tool. No cell like this exists yet.")

    notes(s, """
This is the story the whole company is built around.

Left: a cell running fine. Camera on a post, arm, conveyor, parts.

Middle: at 1:40 a cart clips the camera mount. Nothing alarms, because from
the software's point of view nothing failed. No error code. The robot is still
running and still reporting healthy. But every grab is now about a centimetre
off. In a factory, parts start getting missed around 2:10 and the line stops.
In a company teaching robots, every recording made in that half hour is
quietly wrong and goes straight into the training set.

Then a technician drives out and checks each cause by hand. Camera? Tool?
Joints? Timing? That is hours.

Right: with Kintrace. Flagged at 1:40:23, twenty three seconds in, from the
robot's own logs. A six second check motion confirms it was the camera and
nothing else. New calibration written. Six test grabs. Signed off.

The gap between the middle panel and the right panel is what we sell.
""")


def s04_cost(prs):
    s = blank(prs)
    headline(s, ["You pay for it twice: in ruined training data,",
                 "and in stopped lines."])

    picture(s, "cost_episodes.png", ML, BODY_TOP + 0.18, w=6.5)
    write(box(s, ML, BODY_TOP + 2.28, 6.3, 1.3),
          [("Robot training data", {"bold": True, "size": 16}),
           ("An open audit of 4,959 recorded episodes flagged 15.6% of them "
            "for exclusion, and another half for review.",
            {"size": 14.5, "color": SOFT})], spacing=1.38)

    vrule(s, 8.05, BODY_TOP + 0.10, 3.05)

    write(box(s, 8.5, BODY_TOP + 0.12, 4.2, 0.95), "$2.3M", size=50,
          bold=True, spacing=1.0)
    write(box(s, 8.5, BODY_TOP + 1.10, 4.2, 2.0),
          [("an hour", {"bold": True, "size": 17}),
           ("Up to, for an idle production line at a big car plant. Across "
            "the 500 largest manufacturers, unplanned downtime of every kind "
            "costs $1.4 trillion a year, about 11% of revenue.",
            {"size": 14.5, "color": SOFT})], spacing=1.38)

    source(s, "Episode audit: an open source tool run over 12 public datasets, dev.to, 2026. "
              "Downtime: Siemens, The True Cost of Downtime 2024. Both linked in pitch/SOURCES.md.")

    notes(s, """
Two different buyers feel this, and they feel it differently.

On the left, companies teaching robots. When a robot is physically out of
place it still records, and those recordings look fine. They are labelled with
where the robot thought it was, not where it actually was. Someone audited
nearly five thousand public robot training episodes with an open tool and
marked about one in six as ones you should throw out. I want to be careful:
that tool flags episodes for review, it is not proof, and the author says so.
But at a hundred and eighteen dollars an hour to collect this data, throwing
out one in six is real money.

On the right, factories. An idle line at a big car plant runs to two point
three million dollars an hour.

I am not claiming Kintrace recovers all of that. I am saying the minutes
between a bump and someone noticing are worth a lot, and nobody is selling
those minutes back today.
""")


def s05_why_unsolved(prs):
    s = blank(prs)
    headline(s, ["Everything on the market tells you something is wrong,",
                 "not what moved."])

    cols = [
        ("Monitoring dashboards", "Ember, Foxglove, Formant",
         "They watch the software and the logs. They will tell you a cell is "
         "slow or a run failed. They do not measure where the camera is, so "
         "they cannot name the physical cause."),
        ("Calibration hardware", "Dynalog, CAPTRON, Renishaw",
         "These do measure, and they are accurate. But you bolt a device into "
         "every cell you want covered, and each one mostly does the tool."),
        ("A technician", "How it is actually done",
         "Someone drives out and rules out each cause by hand, one at a time, "
         "with the cell down while they do it. It works. It costs hours."),
    ]

    w = (CW - 1.5) / 3
    for i, (title, who, body) in enumerate(cols):
        x = ML + i * (w + 0.75)
        shape(s, MSO_SHAPE.RECTANGLE, x, BODY_TOP, 1.05, 0.055,
              fill=ACCENT if i == 2 else RULE)
        write(box(s, x, BODY_TOP + 0.28, w, 0.5), title, size=18, bold=True)
        write(box(s, x, BODY_TOP + 0.84, w, 0.35), who, size=12.5, color=FAINT)
        write(box(s, x, BODY_TOP + 1.26, w, 2.4), body, size=14.5, color=SOFT,
              spacing=1.44)

    panel(s, ML, 5.78, CW, 0.86)
    write(box(s, ML + 0.34, 5.98, CW - 0.68, 0.5),
          "We have not found anyone doing all of it from software alone: when "
          "it changed, what moved, by how much, and the fix written back.",
          size=15.5, spacing=1.3)

    source(s, "What each of these does is taken from their own sites and materials. Listed in pitch/SOURCES.md.")

    notes(s, """
Why has nobody done this? They have each done a piece.

Monitoring dashboards. Ember is closest in spirit and their line is literally
"know what changed, know what failed". Foxglove raised forty million last
November. Good products. But they watch the software. They can tell you a run
failed. They cannot tell you the camera is eleven millimetres to the left,
because they never measure where the camera is.

Calibration hardware. Dynalog sells exactly this outcome, restore accuracy
after a crash without re-teaching. It works. The catch is it is hardware you
mount in the cell, and it is mostly about the tool tip. A fleet with two
hundred cells is not buying two hundred fixtures.

Then there is what actually happens most of the time, which is a person
driving out and checking things by hand.

So the gap is the whole sentence: when it changed, what moved, by how much,
and the fix, from software, with no new hardware.
""")


def s06_product(prs):
    s = blank(prs)
    headline(s, "This is the tool's own output, not a mockup.")

    picture(s, "play_record.png", ML, BODY_TOP - 0.04, w=8.55)

    steps = [
        ("1", "Notices", "Finds the moment things changed and lines it up "
                         "with the protective stop the controller logged."),
        ("2", "Explains", "Names the cause and sizes it. Here: the camera, "
                          "18.1 mm, and nothing else moved."),
        ("3", "Fixes", "Writes the new calibration, re-checks, runs six test "
                       "grabs, then signs GO or NO-GO."),
    ]
    x = ML + 8.92
    y = BODY_TOP + 0.06
    for num, title, body in steps:
        shape(s, MSO_SHAPE.OVAL, x, y, 0.34, 0.34, fill=ACCENT, line=INK)
        write(box(s, x, y + 0.06, 0.34, 0.26), num, size=13, bold=True,
              align=PP_ALIGN.CENTER)
        write(box(s, x + 0.48, y + 0.02, 2.7, 0.34), title, size=16, bold=True)
        write(box(s, x + 0.48, y + 0.42, 2.7, 1.2), body, size=13, color=SOFT,
              spacing=1.38)
        y += 1.56

    source(s, "A real run of the tool on our simulated cell. You can run it yourself at the demo link "
              "on the last slide.")

    notes(s, """
This is the record Kintrace printed for the incident on the last slide. It is
the real thing, not a mockup, and you can produce it yourself in the browser
demo in about ten seconds.

Follow the three callouts.

One, it noticed: found the change at 13:40:23 and matched it to the protective
stop already in the controller log.

Two, it explained: camera moved 18.1 millimetres. And critically, it says the
other four things it checked did not move. That is the hard part. A bumped
camera and a drifted joint look almost identical in the logs. The difference
is that a moved camera shifts everything by the same amount, while a joint
that is off gives an error that changes with the arm's pose. Kintrace fits
every explanation and keeps the only one that brings the error down to sensor
noise.

Three, it fixed: wrote the calibration, re-ran the check, six test grabs, GO.

There is no AI model in here. It is geometry and statistics. Same log in, same
answer out, every time, and you can check the arithmetic. For a buyer who has
to certify a cell before it goes back into production, that matters more than
accuracy does.
""")


def s07_proof(prs):
    s = blank(prs)
    headline(s, ["In our simulator, Kintrace named the right cause",
                 "every time. The dashboard baseline we wrote got 69%."])

    picture(s, "results_by_fault.png", 1.97, BODY_TOP + 0.08, h=4.56)

    source(s, "120 runs on our simulated UR5e pick cell, 20 per case, every fault sized to cause the same "
              "20 mm miss. The baseline is ours, built from the signals a typical dashboard tracks. "
              "Never run on a real arm.")

    notes(s, """
Our benchmark. A hundred and twenty runs, twenty for each of five faults plus
twenty healthy runs. Every fault is sized so it causes the same twenty
millimetre miss, so the robot looks equally broken in all of them and the only
question is whether you can tell them apart.

Yellow is Kintrace, grey is the baseline. Look at the top two rows. A joint
knocked off zero: the dashboard gets it right one time in five. A wrong tool
setting loaded: two in five. Those are the expensive ones to diagnose by hand,
and they are exactly where the dashboard is worst, because from its point of
view all of these look like the same thing, which is that grabs are missing.

Two caveats, and I would rather say them than have you find them. One, this is
our simulator. Two, we wrote the baseline as well, so it is not a benchmark
against a shipping product, it is a benchmark against the signals a typical
dashboard tracks.

That is exactly why the first money in this round goes to a real arm.
""")


def s08_early(prs):
    s = blank(prs)
    headline(s, ["It also catches slow drift, about 55 seconds",
                 "before the first grab misses."])

    picture(s, "early_warning.png", 1.80, BODY_TOP + 0.04, h=4.50)

    source(s, "One simulated run of a sagging camera mount. Across 20 drifts it warned before the first "
              "miss 19 times, named the cause right 20 times, and raised 0 false alarms in 230 healthy "
              "windows. Drift is sped up so a run takes minutes. We have not measured lead time at "
              "real-world drift rates.")

    notes(s, """
The last slide was after a crash. This one is the other half, and it is the
half that makes this a subscription rather than a callout fee.

Kintrace reads the logs the robot already writes, with no check motion and no
interruption, and scores the cell every ten seconds. This run is a camera
mount slowly sagging. The score falls. At sixty seconds Kintrace says: your
camera is moving, parts are two and a half millimetres off, at this rate you
start missing in about a minute. At that point nothing has failed yet. The
first missed grab is fifty five seconds later.

Across twenty of these drifts it warned in time nineteen times, named the
cause right all twenty, and gave zero false alarms across two hundred and
thirty healthy windows. The zero matters most. A monitoring tool that cries
wolf gets switched off in a week.
""")


def s09_why_now(prs):
    s = blank(prs)
    headline(s, ["There have never been more robots, and the money",
                 "behind them doubled in three years."])

    picture(s, "why_now.png", ML, BODY_TOP + 0.14, w=7.9)

    x = ML + 8.35
    facts = [
        ("5 million", "industrial robots now operating, a record, up 9% in a "
                      "year. The US installed 38,500 and passed Japan. "
                      "IFR, September 2026."),
        ("47% a year", "forecast growth for physical AI, from $0.89B in 2025 "
                       "to $15.2B by 2032. Every robot in it learns from "
                       "recordings."),
    ]
    y = BODY_TOP + 0.16
    for value, body in facts:
        shape(s, MSO_SHAPE.RECTANGLE, x, y, 0.95, 0.055, fill=ACCENT)
        write(box(s, x, y + 0.24, 3.6, 0.66), value, size=27, bold=True,
              spacing=1.05)
        write(box(s, x, y + 0.92, 3.6, 1.28), body, size=13.5, color=SOFT,
              spacing=1.42)
        y += 2.42

    source(s, "IFR World Robotics 2026. MarketsandMarkets, physical AI. "
              "Venture figures from Robotics Center, State of Robotics 2026. Links in pitch/SOURCES.md.")

    notes(s, """
Why this is the right moment and not two years ago. Three things landed at
once.

First, days ago the IFR published its 2026 report: five million industrial
robots now operating, a record, up nine percent. The US installed thirty eight
and a half thousand and passed Japan to become the second largest market. The
installed base I sell into is at its largest ever and still growing.

Second, the chart. Venture money into robotics went from three point eight
billion in 2022 to nine point four billion in 2025, up forty one percent in
the last year alone. That money buys arms, and every arm it buys is a cell
someone now has to keep straight.

Third, and this is the one I care about most, physical AI is forecast to grow
about forty seven percent a year. That category is companies teaching robots
from recordings. They are the buyer who feels a misaligned camera immediately,
because it silently poisons the thing they are spending all that money to
build.

Two years ago I would have been selling to factories only. Now there is a
second buyer who moves faster and is in more of a hurry.
""")


def s10_market(prs):
    s = blank(prs)
    headline(s, ["At $200 a robot a month, the part we can",
                 "realistically sell to is $1.2B a year."])

    picture(s, "market.png", 1.92, BODY_TOP + 0.02, h=3.84)

    panel(s, ML, 5.96, CW, 0.82)
    write(box(s, ML + 0.34, 6.14, CW - 0.68, 0.5),
          "Top down check: $22.5B was spent on industrial robot services in "
          "2023, heading to $41.6B by 2033. Our $12B bottom up number sits "
          "inside a services market already twice its size.",
          size=14.5, spacing=1.3)

    source(s, "Robot count: IFR World Robotics 2026. Services market: Market.us, October 2024. "
              "The 10% camera share is our own estimate; we looked for a public figure and did not find one.")

    notes(s, """
Bottom up first, because I would rather show the arithmetic than a big number.

Two hundred dollars a robot a month is twenty four hundred a year. Five
million robots is twelve billion a year if every robot on earth bought it.
That is the TAM and it is not a real number. Nobody sells to everybody.

The middle circle is the one I believe. Kintrace needs the robot to have a
camera looking at the work area. My estimate is about one in ten, which gets
to one point two billion a year. I want to flag that the ten percent is my
estimate. I looked for a public figure for what share of the installed robot
base is vision guided and there is not one. The public numbers size the robot
vision market in dollars, which is a different question. So treat that as the
number to push me on.

The small dot is twenty thousand robots five years out, forty eight million a
year. That is the number I am actually trying to hit.

As a check from the other direction: twenty two and a half billion a year is
already spent on industrial robot services, heading to forty one billion. My
bottom up number sits inside a market that already exists and already pays for
this outcome.
""")


def s11_competition(prs):
    s = blank(prs)
    headline(s, "Software that names the cause and fixes it is an empty box.")

    gx, gy, gw, gh = 2.55, BODY_TOP + 0.34, 8.2, 3.56
    panel(s, gx, gy, gw, gh, PANEL)
    # Kintrace owns the top right box, so that box gets the yellow.
    shape(s, MSO_SHAPE.RECTANGLE, gx + gw / 2, gy, gw / 2, gh / 2, fill=ACCENT)
    rule(s, gx, gy + gh / 2, gw, RULE)
    vrule(s, gx + gw / 2, gy, gh, RULE)

    def cell(cx, cy, w, name, line):
        write(box(s, cx, cy, w, 0.34), name, size=14, bold=True)
        write(box(s, cx, cy + 0.32, w, 0.9), line, size=11.5, color=SOFT,
              spacing=1.32)

    cell(gx + 0.32, gy + 0.32, 3.3, "Dynalog, CAPTRON",
         "Finds and fixes it, but you mount hardware in every cell.")
    cell(gx + gw / 2 + 0.32, gy + 0.32, 3.3, "Kintrace",
         "Finds it, sizes it, writes the fix, certifies the cell. "
         "Software only.")
    cell(gx + 0.32, gy + gh / 2 + 0.32, 3.3, "Robot makers",
         "Built-in mastering routines. One brand, run by a technician.")
    cell(gx + gw / 2 + 0.32, gy + gh / 2 + 0.32, 3.3,
         "Ember, Foxglove, dataset checkers",
         "Show you something is off. No geometry, no fix.")

    write(box(s, gx, gy - 0.36, gw, 0.3), "Finds and fixes the cause",
          size=12, color=FAINT, bold=True, align=PP_ALIGN.CENTER)
    write(box(s, gx, gy + gh + 0.10, gw, 0.3), "Only shows a problem",
          size=12, color=FAINT, bold=True, align=PP_ALIGN.CENTER)
    write(box(s, gx - 1.58, gy + gh / 2 - 0.16, 1.46, 0.3), "Needs hardware",
          size=12, color=FAINT, bold=True, align=PP_ALIGN.RIGHT)
    write(box(s, gx + gw + 0.14, gy + gh / 2 - 0.16, 1.44, 0.3),
          "Software only", size=12, color=FAINT, bold=True)

    notes(s, """
Two axes. Left to right is whether you need hardware in the cell. Bottom to
top is whether you find and fix the cause or only flag that something is off.

Bottom right: Ember, Foxglove, dataset quality checkers. Software, easy to
deploy, but they tell you something is wrong.

Top left: Dynalog and CAPTRON. They genuinely fix it. But it is a device per
cell, so it does not scale across a fleet.

Bottom left: the robot makers' own mastering routines. Accurate, one brand
each, and a technician has to run them.

Top right is empty. Software only, and it names the cause and writes the fix.
That is the whole bet, and it is why I am building it now rather than waiting.
""")


def s12_roadmap(prs):
    s = blank(prs)
    headline(s, "Where we are going.")

    stages = [
        ("Now", "The engine works in simulation. A live demo anyone can try, "
                "and the code is open on GitHub."),
        ("3 months", "Kintrace running on a real robot arm, a video of it "
                     "catching and fixing a bump, and first conversations "
                     "with robot teams."),
        ("6 months", "Three paid pilots with robot learning teams and robot "
                     "fleets, and real robot data plugged in through ROS 2 "
                     "logs."),
        ("12 months", "Always-on monitoring for fleets, then working with "
                      "robot makers to ship it on new arms."),
    ]

    track_y = BODY_TOP + 0.82
    rule(s, ML + 0.1, track_y, CW - 0.55, RULE, thick=0.02)

    step = (CW - 0.5) / 4
    for i, (when, body) in enumerate(stages):
        x = ML + 0.1 + i * step
        dot(s, x - 0.09, track_y - 0.085, 0.2,
            fill=ACCENT if i == 0 else PAPER, line=INK)
        write(box(s, x - 0.06, track_y - 0.66, step - 0.4, 0.34), when,
              size=16, bold=True)
        write(box(s, x - 0.06, track_y + 0.34, step - 0.45, 1.8), body,
              size=13.5, color=SOFT, spacing=1.44)

    panel(s, ML, 5.34, CW, 1.10)
    shape(s, MSO_SHAPE.RECTANGLE, ML, 5.34, 0.075, 1.10, fill=ACCENT)
    write(box(s, ML + 0.40, 5.56, CW - 0.8, 0.8),
          [("Where this goes", {"bold": True, "size": 14}),
           ("Every robot arm checked the way a car has a check engine light, "
            "with Kintrace as the standard way to know a robot is physically "
            "right.", {"size": 15, "color": SOFT})], spacing=1.36)

    notes(s, """
Where this goes, in four steps.

Now: the engine works. It runs in simulation, there is a live demo you can
open in a browser and break a robot yourself, and the code is on GitHub. That
is real and you can check all of it today.

Three months: on a real arm. That is the single biggest risk in this company,
that real sensors behave worse than my simulator, so it is the first thing the
money buys. Out of that comes the video I should have had for this meeting,
and the first real conversations with robot teams.

Six months: three paid pilots. Paid, because the only honest test of whether
this is a painkiller is whether someone hands over money. Plus reading real
robot logs in the formats teams already record, which is mostly ROS 2.

Twelve months: always-on monitoring across fleets, and starting conversations
with the arm makers about shipping it on new robots.

The line at the bottom is the actual ambition. Every car has a check engine
light. No robot arm has one. That is what we are building.
""")


def s13_why_back(prs):
    s = blank(prs)
    headline(s, "Why back Kintrace now.")

    points = [
        ("It is built, not an idea",
         "A working engine, a live demo, open code and 32 tests passing. You "
         "can run the whole thing yourself before you decide."),
        ("A hard technical edge",
         "It tells apart faults that look identical from outside, camera "
         "against joint against tool, and gives the exact fix. 120/120 "
         "against 83/120 for a dashboard baseline. That is research, not a "
         "feature someone adds in a week."),
        ("A big market with no software answer",
         "5 million robots, 600,000 added a year, robot learning booming. "
         "Today you get a dashboard that only flags a problem, or hardware "
         "in every cell."),
        ("The founder",
         "Physical AI research at CMU, Harvard Innovation Labs. I built all "
         "of this myself, from idea to working product in weeks."),
    ]

    w = (CW - 0.8) / 2
    for i, (title, body) in enumerate(points):
        col, row = i % 2, i // 2
        x = ML + col * (w + 0.8)
        y = BODY_TOP + row * 2.16
        shape(s, MSO_SHAPE.RECTANGLE, x, y, 0.42, 0.055, fill=ACCENT)
        write(box(s, x, y + 0.24, w - 0.15, 0.4), title, size=17, bold=True)
        write(box(s, x, y + 0.72, w - 0.15, 1.3), body, size=13.5, color=SOFT,
              spacing=1.44)

    source(s, "The 120/120 and 83/120 are from our simulator, and we wrote the baseline. "
              "No customers, no pilots and no real-arm results yet. That is what this round is for.")

    notes(s, """
Four reasons, and I will give you the evidence for each rather than ask you to
take my word.

One, it is built. This is not a deck about something I intend to make. The
engine runs, the demo is live in a browser, the code is on GitHub and there
are thirty two tests passing. You can check every part of that tonight.

Two, the technical edge is real. The hard problem here is that a bumped
camera, a drifted joint and a bent tool look nearly identical from outside.
Everything just misses. Telling them apart takes fitting every explanation to
the geometry and keeping the one that brings the error down to sensor noise.
We get a hundred and twenty out of a hundred and twenty, the dashboard
baseline gets eighty three. In simulation, and I wrote the baseline, but that
is a research result, not a weekend feature.

Three, the market is large, growing, and the existing answers are a dashboard
that only flags a problem or hardware you mount in every cell.

Four, me. I do physical AI research at CMU, I am at the Harvard Innovation
Labs, and I built all of this myself in weeks. I am not going to pretend I
have customers. I have a working product and I move fast.
""")


def s14_ask(prs):
    s = blank(prs)
    headline(s, "$150K takes Kintrace from simulation to paying customers.")

    picture(s, "budget.png", ML - 0.10, BODY_TOP - 0.02, w=7.5)

    x = 8.40
    vrule(s, 8.05, BODY_TOP + 0.02, 3.50)

    write(box(s, x, BODY_TOP, 4.1, 0.34), "What it buys", size=13,
          color=FAINT, bold=True)

    miles = [
        ("3 months", "Kintrace on a real robot arm, a demo video, first user "
                     "conversations"),
        ("6 months", "Three paying pilots"),
        ("12 months", "Results from real customers, ready for a seed round"),
    ]
    y = BODY_TOP + 0.52
    for i, (when, what) in enumerate(miles):
        dot(s, x, y + 0.06, 0.16, fill=ACCENT if i == 0 else PAPER, line=INK)
        write(box(s, x + 0.34, y - 0.02, 3.8, 0.32), when, size=14, bold=True)
        write(box(s, x + 0.34, y + 0.32, 3.8, 0.75), what, size=13,
              color=SOFT, spacing=1.36)
        y += 1.14

    rule(s, ML, 6.36, CW)
    write(box(s, ML, 6.56, 8.0, 0.55),
          [("Rithik Satarla", {"bold": True, "size": 13.5}),
           ("Physical AI research at CMU. Harvard Innovation Labs. "
            "rithiksatarla@gmail.com", {"size": 12, "color": SOFT})],
          spacing=1.34)
    write(box(s, 9.1, 6.56, CW - 8.25, 0.55),
          [("Try it: the browser demo", {"size": 12, "color": SOFT}),
           ("github.com/RithikSatarla/kintrace", {"size": 12, "color": SOFT})],
          spacing=1.34, align=PP_ALIGN.RIGHT)

    notes(s, """
A hundred and fifty thousand. Here is every line of it.

Eighty thousand is a part-time robotics engineer, and I would rather it turned
into a cofounder. This is the honest one: I cannot build the product, sell it,
and run three pilots at the same time. That is the constraint on this company
right now, not the technology.

Twenty thousand is a real industrial arm, a used UR3 or UR5e. Everything I
have shown you is simulation. I need a real arm on a bench that I can bump,
misalign and de-calibrate on purpose, over and over, without asking anyone's
permission.

Thirty thousand is buffer. Hardware slips and pilots take longer than anyone
plans, and I would rather tell you that now than come back for a bridge.

Ten thousand is pilots: travel, plus cameras and printed markers to put on
customer robots.

Five thousand is company setup, legal and accounting. Three thousand is cloud
and software. Two thousand is two or three small desk rigs, about three
hundred dollars of parts each, so I can run faults every day without tying up
the expensive arm.

On the right is what that buys. Three months: running on a real arm, with the
video. Six months: three paying pilots. Twelve months: results from real
customers and a seed round raised on evidence instead of a simulator.
""")


def main():
    prs = deck()
    for fn in (s01_cover, s02_what_we_do, s03_moment, s04_cost, s05_why_unsolved,
               s06_product, s07_proof, s08_early, s09_why_now, s10_market,
               s11_competition, s12_roadmap, s13_why_back, s14_ask):
        fn(prs)
    prs.save(str(OUT))
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
