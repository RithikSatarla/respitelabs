"use client";

import { useEffect, useRef } from "react";
import { jointOrigins, pickPose } from "@/lib/arm";
import type { HeroState } from "@/lib/useHeroLoop";

const INK = "#121212";
const YELLOW = "#FFC400";
const GREY = "#9A9A93";
const RED = "#C62828";
const RULE = "#DEDED8";
const PAPER = "#F7F7F4";

interface Props {
  state: HeroState;
  className?: string;
}

/**
 * The live arm. Real forward kinematics from lib/arm.ts, so the pose on screen
 * is the same maths the diagnosis code runs on.
 *
 * The arm is auto-fitted to the frame across the whole pick cycle, not per
 * frame, so it does not breathe in and out as the arm extends.
 */
export default function ArmCanvas({ state, className }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = cv.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // ---- work out a fixed frame from the whole cycle ----
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (let k = 0; k < 24; k++) {
      for (const p of jointOrigins(pickPose(k / 24))) {
        const x = -(p[0] + p[1] * 0.18);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (p[2] < minZ) minZ = p[2];
        if (p[2] > maxZ) maxZ = p[2];
      }
    }
    minZ = Math.min(minZ, 0);
    const padL = w * 0.30;   // room on the left for the camera post
    const padR = w * 0.24;   // keep the arm clear of the live label column
    const padT = h * 0.13;
    const padB = h * 0.16;
    const scale = Math.min(
      (w - padL - padR) / Math.max(0.01, maxX - minX),
      (h - padT - padB) / Math.max(0.01, maxZ - minZ),
    );
    const ox = padL - minX * scale;
    const floorY = h - padB;
    const oy = floorY + minZ * scale;

    const proj = (p: number[]): [number, number] => [
      ox + -(p[0] + p[1] * 0.18) * scale,
      oy - p[2] * scale,
    ];

    const q = pickPose(state.u);
    const pts = jointOrigins(q).map(proj);

    const jitter = state.shake * 3.4;
    const jx = jitter ? (Math.random() - 0.5) * jitter : 0;
    const jy = jitter ? (Math.random() - 0.5) * jitter : 0;

    // ---- faint dot grid, just enough texture to feel like a readout ----
    const grid = 16;
    ctx.fillStyle = INK;
    ctx.globalAlpha = 0.08;
    for (let gx = grid; gx < w; gx += grid) {
      for (let gy = grid; gy < h; gy += grid) {
        ctx.fillRect(gx - 1, gy - 1, 2, 2);
      }
    }
    ctx.globalAlpha = 1;

    // ---- floor ----
    ctx.strokeStyle = RULE;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.04, floorY);
    ctx.lineTo(w * 0.96, floorY);
    ctx.stroke();

    // ---- table markers on the floor line ----
    ctx.fillStyle = GREY;
    [0.52, 0.78].forEach((f) => ctx.fillRect(w * f, floorY - 4, 14, 4));

    // ---- camera on a post, to the left, looking at the work area ----
    const postX = w * 0.13 + jx;
    const postTop = h * 0.20 + jy;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(postX, floorY);
    ctx.lineTo(postX, postTop);
    ctx.stroke();

    const alerted = state.camOffsetMm > 0;
    ctx.save();
    ctx.translate(postX, postTop);
    ctx.rotate(alerted ? 0.22 : 0);
    ctx.fillStyle = alerted ? YELLOW : INK;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.fillRect(-3, -15, 30, 17);
    ctx.strokeRect(-3, -15, 30, 17);
    ctx.fillStyle = PAPER;
    ctx.beginPath();
    ctx.arc(20, -6.5, 4.6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    // view cone
    ctx.globalAlpha = alerted ? 0.3 : 0.16;
    ctx.fillStyle = YELLOW;
    ctx.beginPath();
    ctx.moveTo(24, -4);
    ctx.lineTo(w * 0.72 - postX, floorY - postTop - 8);
    ctx.lineTo(w * 0.42 - postX, floorY - postTop - 8);
    ctx.closePath();
    ctx.fill();
    ctx.restore();

    // ---- pedestal under the arm ----
    const base = pts[0];
    ctx.fillStyle = INK;
    ctx.fillRect(base[0] - 22, base[1], 44, floorY - base[1]);

    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = INK;
    ctx.lineWidth = 9;
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.stroke();

    pts.forEach(([x, y], i) => {
      ctx.beginPath();
      ctx.arc(x, y, i === 0 ? 9 : 6, 0, Math.PI * 2);
      ctx.fillStyle = PAPER;
      ctx.fill();
      ctx.lineWidth = 2.2;
      ctx.strokeStyle = INK;
      ctx.stroke();
    });

    // ---- the grab at the tool tip ----
    const tip = pts[pts.length - 1];
    if (state.phase === "miss") {
      ctx.strokeStyle = RED;
      ctx.lineWidth = 2.6;
      const s = 8;
      ctx.beginPath();
      ctx.moveTo(tip[0] - s, tip[1] - s);
      ctx.lineTo(tip[0] + s, tip[1] + s);
      ctx.moveTo(tip[0] + s, tip[1] - s);
      ctx.lineTo(tip[0] - s, tip[1] + s);
      ctx.stroke();
    } else {
      ctx.fillStyle = YELLOW;
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.4;
      ctx.fillRect(tip[0] - 8, tip[1] - 5, 16, 11);
      ctx.strokeRect(tip[0] - 8, tip[1] - 5, 16, 11);
    }
  }, [state]);

  return <canvas ref={ref} className={className} aria-hidden="true" />;
}
