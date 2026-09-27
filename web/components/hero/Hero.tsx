"use client";

import { useRef } from "react";
import { motion, useReducedMotion } from "framer-motion";
import ArmCanvas from "./ArmCanvas";
import { useHeroLoop } from "@/lib/useHeroLoop";
import site from "@/data/site.json";

const REPORT =
  `${site.hero.changedAt}  camera moved ${site.hero.cameraMovedMm} mm · fix written · test grabs ${site.hero.testPicks} · ${site.hero.status}`;

export default function Hero() {
  const wrap = useRef<HTMLDivElement>(null);
  const s = useHeroLoop(wrap, REPORT.length);
  const reduce = useReducedMotion();

  const alerted = s.camOffsetMm > 0 && s.phase !== "fixed";
  const labels = [
    { k: "J1", v: "+12.4°" },
    { k: "J3", v: "-48.0°" },
    { k: "CAM Δ", v: alerted ? `${site.hero.cameraMovedMm} mm` : "0.1 mm", hot: alerted },
    { k: "TOOL Δ", v: "0.2 mm" },
    { k: "timing", v: "8 ms" },
  ];

  return (
    <section ref={wrap} className="relative overflow-hidden border-b border-rule">
      <div className="mx-auto grid max-w-[1240px] items-center gap-10 px-5 py-14 sm:px-8 lg:grid-cols-[1.02fr_1fr] lg:gap-14 lg:py-20">
        {/* ---- words ---- */}
        <motion.div
          initial={reduce ? false : { opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <div className="mb-6 inline-flex items-center gap-2 border border-rule bg-panel px-2.5 py-1">
            <span className="relative flex h-2 w-2">
              {!reduce && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-go opacity-60" />
              )}
              <span className="relative inline-flex h-2 w-2 rounded-full bg-go" />
            </span>
            <span className="font-mono text-[11px] tracking-tight text-soft">
              Demo mode · {site.version}
            </span>
          </div>

          <h1 className="text-[clamp(2rem,5vw,3.6rem)] font-extrabold leading-[1.02] tracking-[-0.025em]">
            Your robot got bumped.
            <br />
            Kintrace already knows.
          </h1>

          <p className="mt-5 max-w-[46ch] text-[clamp(1rem,1.35vw,1.2rem)] leading-relaxed text-soft">
            It finds what moved on a robot arm, fixes it, and tells you it is
            good to go. From data the robot already records.
          </p>

          <div className="mt-8 flex flex-wrap gap-3">
            <a
              href="/play/"
              className="bg-accent px-6 py-3.5 text-sm font-bold text-ink transition hover:brightness-95 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
            >
              Try it live
            </a>
            <a
              href="mailto:rithiksatarla@gmail.com?subject=Kintrace"
              className="border border-ink px-6 py-3.5 text-sm font-bold text-ink transition hover:bg-ink hover:text-paper focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ink"
            >
              Book a 15 min call
            </a>
          </div>

          {/* stats row, like a spec sheet */}
          <dl className="mt-10 grid grid-cols-3 gap-4 border-t border-rule pt-6">
            {[
              [site.stats.robotsWorldwide, site.stats.robotsLabel],
              [site.stats.faultsFound, site.stats.faultsLabel],
              [`${site.stats.onsetSeconds} s`, site.stats.onsetLabel],
            ].map(([v, l]) => (
              <div key={l}>
                <dt className="font-mono text-[clamp(1.1rem,2vw,1.6rem)] font-semibold leading-none">
                  {v}
                </dt>
                <dd className="mt-2 text-[11px] leading-snug text-faint">{l}</dd>
              </div>
            ))}
          </dl>
        </motion.div>

        {/* ---- the live cell ---- */}
        <div className="relative">
          <div className="relative aspect-[4/3] w-full border border-rule bg-panel">
            <ArmCanvas state={s} className="h-full w-full" />

            {/* live labels */}
            <ul className="pointer-events-none absolute right-3 top-3 space-y-1 text-right">
              {labels.map((l) => (
                <li key={l.k} className="font-mono text-[10px] sm:text-[11px]">
                  <span className="text-faint">{l.k} </span>
                  <span
                    className={
                      l.hot
                        ? "bg-accent px-1 font-semibold text-ink"
                        : "text-soft"
                    }
                  >
                    {l.v}
                  </span>
                </li>
              ))}
            </ul>

            {/* the report types out */}
            {(s.phase === "report" || s.phase === "fixed") && (
              <div className="absolute bottom-3 left-3 right-3 bg-ink px-3 py-2">
                <p className="font-mono text-[10px] leading-snug text-paper sm:text-[11.5px]">
                  {REPORT.slice(0, s.typed)}
                  {s.typed < REPORT.length && (
                    <span className="ml-0.5 inline-block h-3 w-[6px] translate-y-[2px] bg-accent" />
                  )}
                </p>
              </div>
            )}
          </div>
          <p className="mt-2 font-mono text-[10px] text-faint">
            Live render. Numbers from a saved run in incidents/. Simulation.
          </p>
        </div>
      </div>
    </section>
  );
}
