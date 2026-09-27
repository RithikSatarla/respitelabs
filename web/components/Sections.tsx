"use client";

import { motion, useReducedMotion } from "framer-motion";
import type { ReactNode } from "react";
import site from "@/data/site.json";

/** Fade a block up as it scrolls in. Does nothing under reduced motion. */
export function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y: 18 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.45, delay, ease: "easeOut" }}
    >
      {children}
    </motion.div>
  );
}

export function Section({
  id,
  kicker,
  title,
  children,
  tight,
}: {
  id?: string;
  kicker?: string;
  title?: string;
  children: ReactNode;
  tight?: boolean;
}) {
  return (
    <section id={id} className="border-b border-rule">
      <div className={`mx-auto max-w-[1240px] px-5 sm:px-8 ${tight ? "py-12" : "py-16 lg:py-20"}`}>
        {kicker && (
          <Reveal>
            <p className="mb-3 font-mono text-[11px] uppercase tracking-[0.08em] text-faint">
              {kicker}
            </p>
          </Reveal>
        )}
        {title && (
          <Reveal delay={0.04}>
            <h2 className="mb-9 max-w-[24ch] text-[clamp(1.5rem,3vw,2.35rem)] font-bold leading-[1.15] tracking-[-0.02em]">
              {title}
            </h2>
          </Reveal>
        )}
        {children}
      </div>
    </section>
  );
}

/** The scrolling tail of checks across demo cells. */
export function FeedStrip() {
  const reduce = useReducedMotion();
  const row = site.feed.concat(site.feed);
  return (
    <div className="overflow-hidden border-b border-rule bg-panel py-2.5">
      <div
        className={`flex w-max gap-7 whitespace-nowrap ${reduce ? "" : "animate-[slide_46s_linear_infinite]"}`}
        style={
          reduce
            ? undefined
            : ({ animationName: "slide" } as React.CSSProperties)
        }
      >
        {row.map((f, i) => (
          <span key={i} className="font-mono text-[11px] text-soft">
            <span className="text-faint">{f.cell}</span>{" "}
            {f.text}{" "}
            <span
              className={
                f.state === "go"
                  ? "font-semibold text-go"
                  : f.state === "nogo"
                    ? "font-semibold text-nogo"
                    : "text-faint"
              }
            >
              {f.state === "go" ? "GO" : f.state === "nogo" ? "NO-GO" : "ok"}
            </span>
          </span>
        ))}
      </div>
      <style>{`@keyframes slide{from{transform:translateX(0)}to{transform:translateX(-50%)}}`}</style>
    </div>
  );
}

export function Measured() {
  const m = site.measured;
  const items = [
    [m.incidents, "incidents got the right cause and the right GO"],
    [m.faults, `faults named right, against ${m.baseline} for a dashboard baseline`],
    [m.driftsEarly, "slow drifts caught before the first missed grab"],
    [`${m.falseAlarms}`, `false alarms in ${m.healthyWindows} healthy windows`],
  ];
  return (
    <>
      <div className="grid gap-x-8 gap-y-9 sm:grid-cols-2 lg:grid-cols-4">
        {items.map(([v, l], i) => (
          <Reveal key={l} delay={i * 0.05}>
            <div className="border-t-2 border-accent pt-4">
              <p className="font-mono text-[clamp(1.6rem,3.2vw,2.4rem)] font-semibold leading-none">
                {v}
              </p>
              <p className="mt-3 text-sm leading-snug text-soft">{l}</p>
            </div>
          </Reveal>
        ))}
      </div>
      <Reveal delay={0.2}>
        <p className="mt-9 text-xs text-faint">
          All in simulation so far, on a UR5e-class cell we built, and we wrote
          the dashboard baseline too. Real arm next. Raw numbers are in{" "}
          <span className="font-mono">bench_out/</span>.
        </p>
      </Reveal>
    </>
  );
}
