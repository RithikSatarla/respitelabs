# Speaker notes

What to say on each slide, in the words you would actually use. These are
also inside the pptx, under each slide, so they show up in presenter view.

Generated from the deck by `python pitch/export_notes.py`. Edit the notes in
`pitch/build_deck.py` and rebuild, or edit them in PowerPoint and they stay.

## Slide 1. Kintrace

Hi, I am Rithik. Kintrace is physical observability for robot arms.

Here is the one line. A robot is a machine that believes things about itself.
It believes where its camera is, how long its gripper is, where its joints are
zeroed. When one of those beliefs stops being true, the robot does not know.
It keeps working, and it keeps getting it wrong.

Kintrace is the thing that checks those beliefs against reality, and when one
is wrong, tells you which one, by how much, and writes the correction back.

If you saw this a few months ago it was called LoopCell. Same product.

## Slide 2. At 1:40 a cart clips a camera. Nobody finds out until 2:10.

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

## Slide 3. You pay for it twice: in ruined training data, and in stopped lines.

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

## Slide 4. Everything on the market tells you something is wrong, not what moved.

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

## Slide 5. Kintrace reads the robot's own logs, then says what moved and writes the fix.

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

## Slide 6. In our simulator, Kintrace named the right cause every time. The dashboard baseline we wrote got 69%.

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

## Slide 7. It also catches slow drift, about 55 seconds before the first grab misses.

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

## Slide 8. There have never been more robots, and the money behind them doubled in three years.

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

## Slide 9. At $200 a robot a month, the part we can realistically sell to is $1.2B a year.

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

## Slide 10. $200 a robot a month, against a technician visit that costs $1,500 to $3,000.

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

## Slide 11. Raising [$150K] to put Kintrace on real arms and into three paid pilots.

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
