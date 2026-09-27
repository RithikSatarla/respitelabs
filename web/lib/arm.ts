/**
 * Forward kinematics for the UR5e-class arm, ported from kintrace/robot.py.
 *
 * Same standard DH numbers, published by Universal Robots. Keep these in sync
 * with DH_A / DH_D / DH_ALPHA in kintrace/robot.py.
 */

export const DH_A = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0];
export const DH_D = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996];
export const DH_ALPHA = [Math.PI / 2, 0.0, 0.0, Math.PI / 2, -Math.PI / 2, 0.0];

export const HOME_Q = [0.0, -1.9, 1.9, -1.57, -1.57, 0.0];
export const JOINT_NAMES = ["base", "shoulder", "elbow", "wrist 1", "wrist 2", "wrist 3"];

export type Mat4 = number[]; // row major, length 16
export type Vec3 = [number, number, number];

function identity(): Mat4 {
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

function mul(a: Mat4, b: Mat4): Mat4 {
  const out = new Array(16).fill(0) as Mat4;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[r * 4 + k] * b[k * 4 + c];
      out[r * 4 + c] = s;
    }
  }
  return out;
}

/** One standard-DH link transform. */
function dhLink(theta: number, d: number, a: number, alpha: number): Mat4 {
  const ct = Math.cos(theta);
  const st = Math.sin(theta);
  const ca = Math.cos(alpha);
  const sa = Math.sin(alpha);
  return [
    ct, -st * ca, st * sa, a * ct,
    st, ct * ca, -ct * sa, a * st,
    0, sa, ca, d,
    0, 0, 0, 1,
  ];
}

/** Origin of every joint frame, base first, flange last. Metres. */
export function jointOrigins(q: number[]): Vec3[] {
  let T = identity();
  const pts: Vec3[] = [[0, 0, 0]];
  for (let i = 0; i < 6; i++) {
    T = mul(T, dhLink(q[i], DH_D[i], DH_A[i], DH_ALPHA[i]));
    pts.push([T[3], T[7], T[11]]);
  }
  return pts;
}

/** Flange pose origin only. */
export function toolPoint(q: number[]): Vec3 {
  const pts = jointOrigins(q);
  return pts[pts.length - 1];
}

/**
 * A slow pick loop, shaped like the sim's pick motion: reach out over the
 * conveyor, dip, close, lift, return. `u` runs 0..1.
 */
export function pickPose(u: number): number[] {
  const tau = Math.PI * 2;
  const swing = Math.sin(u * tau) * 0.55;
  const dip = Math.max(0, Math.sin(u * tau * 2)) * 0.34;
  return [
    HOME_Q[0] + swing,
    HOME_Q[1] + dip * 0.7,
    HOME_Q[2] - dip * 0.5,
    HOME_Q[3] - dip * 0.25,
    HOME_Q[4],
    HOME_Q[5] + swing * 0.4,
  ];
}

/** Project a metre-space point to screen, viewed from the side. */
export function project(
  p: Vec3,
  w: number,
  h: number,
  scale: number,
  ox: number,
  oy: number,
): [number, number] {
  // x forward, z up. y (sideways) is squashed in slightly for depth.
  const x = p[0] + p[1] * 0.18;
  const z = p[2];
  return [ox + x * scale, oy - z * scale];
}
