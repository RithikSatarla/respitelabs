# Speaker notes

What to say on each slide, in the words you would actually use. These are
also inside the pptx, under each slide, so they show up in presenter view.

Generated from the deck by `python pitch/export_notes.py`. Edit the notes in
`pitch/build_deck.py` and rebuild, or edit them in PowerPoint and they stay.

## Slide 1. Kintrace

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

## Slide 2. Kintrace finds what moved on a robot, fixes it, and tells you it is good to go.

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

## Slide 3. At 1:40 a cart clips a camera. Nobody finds out until 2:10.

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

## Slide 4. You pay for it twice: in ruined training data, and in stopped lines.

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

## Slide 5. Everything on the market tells you something is wrong, not what moved.

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

## Slide 6. This is the tool's own output, not a mockup.

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

## Slide 7. In our simulator, Kintrace named the right cause every time. The dashboard baseline we wrote got 69%.

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

## Slide 8. It also catches slow drift, about 55 seconds before the first grab misses.

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

## Slide 9. There have never been more robots, and the money behind them doubled in three years.

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

## Slide 10. At $200 a robot a month, the part we can realistically sell to is $1.2B a year.

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

## Slide 11. Software that names the cause and fixes it is an empty box.

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

## Slide 12. Where we are going.

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

## Slide 13. Why back Kintrace now.

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

## Slide 14. $150K takes Kintrace from simulation to paying customers.

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
