# Competitors (Sept 2026)

Short version: monitoring tools tell you something is wrong. Calibration
hardware fixes one thing if you bolt a device into every cell. Nobody does
all of it from software: when it changed, what moved, the fix, and a signed
GO / NO-GO.

| Company | What they do | Money | Where they stop |
|---|---|---|---|
| Ember Robotics (YC S24) | Logs, telemetry and release tracking for teams building hardware. "Know what changed. Know what failed." | about $500K (Tracxn) | Watches the software. Doesn't name the physical cause, move the robot, write a fix or sign off a cell. |
| Foxglove | Data and visualization platform for robotics developers. Customers include Amazon, NVIDIA, Dexterity. | $40M Series B, Nov 2025 ($58M+ total) | Built for engineers debugging, not for ops teams getting a cell back up. |
| Formant, InOrbit | Fleet management and remote ops dashboards. | venture backed | Tell you a robot is down or slow. Don't say what moved. |
| Roboto AI | Search and analysis over robot log data. | venture backed | Finds events in logs. No geometry, no fix. |
| Dynalog (DynaCal, RoPOD, AutoCal) | Calibration systems, including TCP and mastering recovery after a crash. Ford case study. | established company | Closest to us. Needs hardware per cell, focused on tool and joint mastering. No camera, no timing, no always-on watch. |
| CAPTRON, advintec, Renishaw | TCP measurement fixtures. 5 to 15 s per check. | established | Tool only. A fixture in every cell. |
| Robot makers (FANUC, UR, ABB) | Built-in calibration and mastering routines. | n/a | Manual, one brand each, done by a technician. |
| Open source (easy_handeye and similar) | Hand-eye calibration from scratch. | free | You redo the whole calibration by hand. Doesn't tell you what changed. |

## What this means for us

1. Dynalog proves people already pay to recover accuracy after a crash. Study them. Their pitch "restore accuracy without re-teaching" is our pitch too, minus the hardware.
2. Ember and Foxglove prove people pay for robot software tools. Foxglove raised $40M. We are not a dashboard, so we can sit next to them.
3. Our line: software only, every cause (camera, tool, joint zero, timing), when + what + fix + signed GO, and a health score that warns before misses.
4. The risk: a robot maker or Dynalog adds a software version. Move fast on a real-arm video and 2 or 3 paying fleets.

Sources:
- https://www.emberrobotics.com/
- https://tracxn.com/d/companies/ember-robotics/__KRMVWV-gKfvSrc42kRWEHFVFqbW73f6Dgb5OnblQet0/funding-and-investors
- https://www.therobotreport.com/foxglove-raises-40m-scale-data-platform-roboticists/
- https://dynalog-us.com/products/
- https://dynalog-us.com/blog/robot-crash-recovery-restore-accuracy-without-reteaching
- https://factory-automation-machinery.bizlinktech.com/news/article/robot-tool-center-point-system-an-unsung-hero-in-avoiding-downtime-and-sunk-costs/
- https://github.com/IFL-CAMP/easy_handeye
