# Sources for the Kintrace deck

Every number on a slide is listed here. Checked on 26 September 2026 by opening
each page. If a number is not in the "Checked and confirmed" list, it is our own
estimate or our own measurement, and the deck says so on the slide.

## Checked and confirmed

**Robots in the world.** 5 million industrial robots operating worldwide in 2025,
up 9%. More than 600,000 installed in 2025, up 11%. The United States installed
almost 38,500, up 12%, and passed Japan to become the second largest market after
China. Installations are forecast to reach 655,000 in 2026 and 806,000 in 2029.
IFR World Robotics 2026, released 24 September 2026.
https://ifr.org/ifr-press-releases/news/five-million-robots-now-operate-in-factories-globally

**Cost of downtime.** The 500 biggest companies in the world lose about $1.4
trillion a year to unplanned downtime, about 11% of their revenue. An idle
production line in a major automotive plant costs up to $2.3 million an hour. The
average large plant loses 27 hours a month to unplanned downtime, down from 39
hours in 2019. Siemens, The True Cost of Downtime 2024, reported by the AEMT.
https://www.theaemt.com/resource/the-true-cost-of-downtime-2024-a-comprehensive-analysis.html

**Industrial robotics services market.** $22.5 billion in 2023, forecast to
$41.6 billion by 2033, a 6.35% CAGR. Market.us, published October 2024.
https://market.us/report/industrial-robotics-services-market/

**Physical AI market.** $0.89 billion in 2025, forecast to $15.24 billion by
2032, a 47.2% CAGR over 2026 to 2032. MarketsandMarkets.
https://www.marketsandmarkets.com/Market-Reports/physical-ai-market-240269196.html

**Robotics venture funding and data cost.** Venture investment in robotics
reached $9.4 billion globally in 2025, up 41% on 2024. The fully loaded cost of
an hour of teleoperation data fell from about $340 an hour in early 2024 to $118
an hour as of March 2026. Robotics Center, State of Robotics 2026. Note this is
a report by a company that sells robot hardware and data, not a neutral body.
https://www.roboticscenter.ai/state-of-robotics-2026

**Messy robot training data.** An open source audit tool was run over 4,959
episodes from 12 robot datasets. It marked 775 episodes, 15.6%, as EXCLUDE, and
another 2,473, 49.9%, as REVIEW. The author says plainly that there was no
independent human ground truth, so these are signals to investigate and not
proof. This is one person's blog post, not a peer reviewed study.
https://dev.to/liesliy/robot-training-data-is-messier-than-you-think-auditing-4959-episodes-with-an-open-source-tool-20ke

**Foxglove funding.** $40 million Series B announced 12 November 2025, led by
Bessemer Venture Partners. More than $58 million raised since 2021.
https://www.therobotreport.com/foxglove-raises-40m-scale-data-platform-roboticists/

**Ember Robotics.** Their own site says "Know what changed. Know what failed."
and describes tracing failures and releases across hardware systems.
https://www.emberrobotics.com/
The YC S24 batch and the roughly $500K raised come from Tracxn, which we have
not checked against a second source.
https://tracxn.com/d/companies/ember-robotics/__KRMVWV-gKfvSrc42kRWEHFVFqbW73f6Dgb5OnblQet0/funding-and-investors

**Dynalog.** Sells roPOD and AccuBeam, which continuously monitor a robot and its
tool center point for position changes from wear and from crashes. It is mounted
hardware in the cell.
https://dynalog-us.com/blog/robot-crash-recovery-restore-accuracy-without-reteaching

## Our own measurements, from this repo

These come from running the code in this repo. They are simulation, not a real
robot. The raw files are in `bench_out/`.

- 120 runs, right cause named 100% of the time, against 69% for the dashboard
  baseline we wrote. Per fault numbers on the results slide. `bench_out/bench.json`
- 50 of 50 incidents got the right cause and the right GO or NO-GO.
  `bench_out/incident_bench.json`
- 19 of 20 slow drifts flagged before the first missed pick, 0 false alarms in
  230 healthy windows. `bench_out/watch_bench.json`
- The early warning chart is one run, camera sag, seed 3. Warning at 60 seconds,
  first missed pick at 115 seconds. `bench_out/watch_camera_sag_seed3.json`

The baseline we compare against is one we built ourselves, from the numbers a
typical robot dashboard tracks. We are not comparing against a real product.

## Estimates, not sourced

The deck marks each of these as an estimate.

- **10% of robots have a camera.** Our estimate, used for the SAM. We looked for
  a public figure for the share of the installed robot base that is vision
  guided and did not find one. The public numbers we found size the robot vision
  market in dollars, which is a different thing. This is the softest number on
  the deck and the slide says so.
- **A technician visit costs about $1,500 to $3,000.** Our estimate from talking
  to people. Not sourced.
- **$200 per robot a month, and about $300 per fixed incident in pilots.** Our
  intended pricing, not a market rate.
- **20,000 robots in 5 years.** Our plan, not a forecast from anyone else.

## A claim we removed

The README and an earlier draft of this deck said "real drift is slower, so the
warning comes earlier". We checked it against our own 20 drift runs in
`bench_out/watch_bench.json` and it does not hold. The correlation between drift
speed and warning lead time is +0.204, and the slower half of the runs had a
slightly shorter median lead, 46.3 s against 51.3 s. The warning fires at a
roughly fixed error, about 2.5 mm, not at a fixed time, so the reasoning behind
the claim was wrong. We have removed it from the deck and from the README. We
have not measured lead time at real-world drift rates and now say so.

## Numbers we dropped

None of the numbers we started with failed a check. The only change is that the
IFR installation growth is 11%, not 9%. The 9% figure is the growth in the total
number of robots operating, which is what the deck uses.
