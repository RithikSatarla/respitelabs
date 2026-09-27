import type { NextConfig } from "next";

// Static export, so the whole site is plain files. That keeps the hosting
// choice open: Cloudflare Pages or Netlify can serve it straight from the
// private repo, with no Node runtime and nothing to pay for.
const nextConfig: NextConfig = {
  output: "export",
  // Next 16 writes its own AGENTS.md / CLAUDE.md next to the app. This repo
  // already has one at the root, so keep it from adding more.
  agentRules: false,
  // The headless browser we screenshot with hits 127.0.0.1, and dev blocks
  // that by default, which stops the dev runtime loading and breaks hydration.
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
