# Kintrace

**Physical observability for robot arms.** Ember watches the code. Kintrace watches the body.

**Problem.** After a crash or a change, a robot arm starts missing. Five different things could be wrong: camera bumped, gripper bent, tool offset edited, a joint's zero off after service, or commands arriving late. They all look the same. Someone drives out and checks each one by hand. An idle automotive line costs up to $2.3M an hour, and large plants lose about 27 hours a month to unplanned downtime (Siemens, 2024).

**Product.**
1. Find when it changed, from the robot's own logs.
2. Check what moved with a short check motion (6.3 s in simulation).
3. Write the fix to the controller.
4. Certify: re-check, 6 test picks, signed GO / NO-GO.
5. Watch: a health score that warns before picks fail.

Software only, no fixtures. The checks are geometry and statistics, so the same log always gives the same answer; a small vision model only finds the gripper in each image.

**Proof so far (simulation).** 50/50 incidents right. 120/120 faults named vs 83/120 for a dashboard baseline. 19/20 slow drifts flagged before the first missed pick.

**Real robot recordings (DROID, Franka, held-out cameras).** Gripper found to 19.9 px median. On 58 cameras whose calibration DROID replaced, the fix moves the camera from 153 mm to 79 mm (median) from DROID's corrected pose. Small camera bumps (2 cm) are not caught reliably on this data yet. Real arm: test next, nothing measured yet.

**Customers.** Robots-as-a-service and picking fleets first, since they pay for every technician visit. Then robot makers.

**Price.** $200 per robot a month, or about $300 per recovered incident in pilots.

**Ask.** Raising [$150K] pre-seed for a real-arm proof and 3 paid pilots.

Rithik Satarla · rithiksatarla@gmail.com
