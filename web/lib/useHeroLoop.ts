"use client";

import { useEffect, useRef, useState } from "react";

export type Phase = "normal" | "bump" | "miss" | "report" | "fixed";

export interface HeroState {
  t: number;        // seconds since the loop started
  u: number;        // 0..1 through the current pick
  phase: Phase;
  camOffsetMm: number;
  shake: number;    // 0..1, decays right after the bump
  typed: number;    // characters of the report revealed
}

/** How long each phase of the 12 second loop lasts, in seconds. */
const T_BUMP = 5.0;
const T_MISS = 6.2;
const T_REPORT = 7.0;
const T_FIXED = 10.6;
const T_LOOP = 12.0;

/**
 * Drives the bump-and-fix loop.
 *
 * Pauses when the tab is hidden or the hero scrolls out of view, so it is not
 * burning frames in the background. Under prefers-reduced-motion it does not
 * animate at all and parks on the "fixed, GO" frame.
 */
export function useHeroLoop(ref: React.RefObject<HTMLElement | null>, reportLength: number) {
  const [state, setState] = useState<HeroState>({
    t: 0, u: 0, phase: "normal", camOffsetMm: 0, shake: 0, typed: 0,
  });
  const [reduced, setReduced] = useState(false);
  const raf = useRef<number | null>(null);
  const visible = useRef(true);
  const onScreen = useRef(true);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  useEffect(() => {
    if (reduced) {
      // Still frame of the moment that matters: fixed, and signed off.
      setState({ t: T_FIXED, u: 0.5, phase: "fixed", camOffsetMm: 0, shake: 0, typed: reportLength });
      return;
    }

    const onVis = () => { visible.current = !document.hidden; };
    document.addEventListener("visibilitychange", onVis);

    let io: IntersectionObserver | null = null;
    if (ref.current) {
      io = new IntersectionObserver(
        (entries) => { onScreen.current = entries[0]?.isIntersecting ?? true; },
        { threshold: 0.05 },
      );
      io.observe(ref.current);
    }

    let start: number | null = null;
    let last = 0;

    const tick = (now: number) => {
      raf.current = requestAnimationFrame(tick);
      if (!visible.current || !onScreen.current) { start = null; return; }
      if (start === null) { start = now - last * 1000; }

      const t = ((now - start) / 1000) % T_LOOP;
      last = t;

      let phase: Phase = "normal";
      if (t >= T_FIXED) phase = "fixed";
      else if (t >= T_REPORT) phase = "report";
      else if (t >= T_MISS) phase = "miss";
      else if (t >= T_BUMP) phase = "bump";

      const sinceBump = t - T_BUMP;
      const shake = phase === "normal" || phase === "fixed"
        ? 0
        : Math.max(0, 1 - sinceBump / 0.55);

      const camOffsetMm = t < T_BUMP ? 0 : (phase === "fixed" ? 0 : 18.1);

      const typed = t < T_REPORT
        ? 0
        : Math.min(reportLength, Math.round(((t - T_REPORT) / 1.5) * reportLength));

      setState({ t, u: (t / 4.0) % 1, phase, camOffsetMm, shake, typed });
    };

    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
      document.removeEventListener("visibilitychange", onVis);
      io?.disconnect();
    };
  }, [ref, reduced, reportLength]);

  return { ...state, reduced };
}
