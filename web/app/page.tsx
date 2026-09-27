import Hero from "@/components/hero/Hero";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import { Section, Reveal, FeedStrip, Measured } from "@/components/Sections";

const STEPS = [
  {
    n: "1",
    title: "Notices",
    body:
      "It finds the moment the physical setup changed, from data the robot already writes, and lines it up with the event the controller logged.",
    out: "Changed at 13:40:23\nlines up with protective stop",
  },
  {
    n: "2",
    title: "Explains",
    body:
      "It names the cause and sizes it. Not an alert. A measurement you can act on, and it says what did not move.",
    out: "Camera moved: 18.1 mm, 0.04 deg\nparts appear 18 mm off",
  },
  {
    n: "3",
    title: "Fixes",
    body:
      "It writes the corrected calibration back, runs the check again, does six test grabs, then signs the cell GO or NO-GO.",
    out: "test picks: 6/6 ok\nSTATUS: GO",
  },
];

const TODAY = [
  ["1:40", "A cart clips the camera mount. Nothing alarms."],
  ["1:40 on", "Every grab is a centimetre off. The robot reports healthy."],
  ["2:10", "Grabs start missing. The line stops."],
  ["Hours later", "A technician checks each possible cause by hand."],
];

const WITH = [
  ["1:40:23", "Flagged. The camera moved 18.1 mm."],
  ["1:40:29", "A 6 second check confirms nothing else moved."],
  ["1:41", "The new calibration is written to the controller."],
  ["1:43", "Six test grabs. Signed off, back in production."],
];

const QUAD = [
  ["Dynalog, CAPTRON", "Finds and fixes it, but you mount hardware in every cell.", false],
  ["Kintrace", "Finds it, sizes it, writes the fix, certifies the cell. Software only.", true],
  ["Robot makers", "Built-in mastering routines. One brand, run by a technician.", false],
  ["Ember, Foxglove, dataset checkers", "Show you something is off. No geometry, no fix.", false],
];

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <FeedStrip />

        <div className="border-b border-rule">
          <p className="mx-auto max-w-[1240px] px-5 py-4 text-xs text-faint sm:px-8">
            Built by a CMU physical AI research intern · Harvard Innovation Labs ·
            Open source on GitHub
          </p>
        </div>

        <Section
          id="product"
          kicker="How it works"
          title="Notices, explains, fixes. In that order."
        >
          <div className="grid gap-8 lg:grid-cols-3">
            {STEPS.map((s, i) => (
              <Reveal key={s.n} delay={i * 0.06}>
                <div className="flex h-full flex-col">
                  <div className="mb-4 flex items-center gap-3">
                    <span className="grid h-7 w-7 place-items-center border border-ink bg-accent text-xs font-bold">
                      {s.n}
                    </span>
                    <h3 className="text-lg font-bold">{s.title}</h3>
                  </div>
                  <p className="mb-5 text-[15px] leading-relaxed text-soft">{s.body}</p>
                  <pre className="mt-auto whitespace-pre-wrap border border-rule bg-panel p-3.5 font-mono text-[11.5px] leading-relaxed text-ink">
                    {s.out}
                  </pre>
                </div>
              </Reveal>
            ))}
          </div>
        </Section>

        <Section
          id="story"
          kicker="An example, from our simulator"
          title="At 1:40 a cart clips a camera. Nobody finds out until 2:10."
        >
          <div className="grid gap-10 lg:grid-cols-2">
            {[
              { label: "Today", rows: TODAY, hot: false },
              { label: "With Kintrace", rows: WITH, hot: true },
            ].map((lane) => (
              <Reveal key={lane.label}>
                <p
                  className={`mb-5 inline-block px-2 py-0.5 text-sm font-bold ${lane.hot ? "bg-accent text-ink" : "text-soft"}`}
                >
                  {lane.label}
                </p>
                <ol className="border-l border-rule">
                  {lane.rows.map(([when, what]) => (
                    <li key={when} className="relative py-3.5 pl-6">
                      <span
                        className={`absolute -left-[5px] top-[19px] h-2.5 w-2.5 rounded-full border border-ink ${lane.hot ? "bg-accent" : "bg-paper"}`}
                      />
                      <p className="font-mono text-xs font-semibold">{when}</p>
                      <p className="mt-1 text-[15px] leading-snug text-soft">{what}</p>
                    </li>
                  ))}
                </ol>
              </Reveal>
            ))}
          </div>
          <Reveal delay={0.16}>
            <p className="mt-8 text-xs text-faint">
              An illustration, drawn from a run on our simulator. The 18.1 mm and
              the timings are printed by the tool. No cell like this exists yet.
            </p>
          </Reveal>
        </Section>

        <Section id="demo" kicker="Crash it yourself" title="Break it. Watch it get fixed.">
          <Reveal>
            <p className="mb-7 max-w-[62ch] text-[15px] leading-relaxed text-soft">
              Pick a fault, choose how big, and break the cell. Kintrace runs on
              the result and prints the same record it would in production. Every
              number is a real run of the engine in your browser, nothing typed
              by hand.
            </p>
            <div className="flex flex-wrap gap-3">
              <a
                href="/play/"
                className="bg-accent px-6 py-3.5 text-sm font-bold text-ink transition hover:brightness-95"
              >
                Open the demo
              </a>
              <a
                href="/demo/"
                className="border border-ink px-6 py-3.5 text-sm font-bold transition hover:bg-ink hover:text-paper"
              >
                See a worked incident
              </a>
            </div>
          </Reveal>
        </Section>

        <Section
          id="measured"
          kicker="Measured, not claimed"
          title="What it scores when we run it."
        >
          <Measured />
        </Section>

        <Section id="fit" kicker="Where we fit" title="Software that names the cause and fixes it is an empty box.">
          <Reveal>
            <div className="grid grid-cols-2 border border-rule">
              {QUAD.map(([name, line, win]) => (
                <div
                  key={name as string}
                  className={`border-b border-r border-rule p-5 last:border-r-0 sm:p-7 ${win ? "bg-accent" : ""}`}
                >
                  <h3 className="mb-2 text-sm font-bold sm:text-base">{name}</h3>
                  <p className={`text-[13px] leading-snug ${win ? "text-ink/75" : "text-soft"}`}>
                    {line}
                  </p>
                </div>
              ))}
            </div>
            <p className="mt-4 font-mono text-[11px] text-faint">
              across: needs hardware in the cell → software only ·
              down: finds and fixes the cause → only shows a problem
            </p>
          </Reveal>
        </Section>

        <section className="bg-ink">
          <div className="mx-auto flex max-w-[1240px] flex-col items-start gap-7 px-5 py-16 sm:px-8 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="max-w-[18ch] text-[clamp(1.6rem,3.4vw,2.6rem)] font-bold leading-[1.1] tracking-[-0.02em] text-paper">
                Try it on your robot.
              </h2>
              <p className="mt-3 font-mono text-sm text-paper/70">
                rithiksatarla@gmail.com
              </p>
            </div>
            <a
              href="mailto:rithiksatarla@gmail.com?subject=Kintrace"
              className="bg-accent px-7 py-4 text-sm font-bold text-ink transition hover:brightness-95"
            >
              Send us a crash
            </a>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
