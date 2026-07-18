import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "var(--bg-base)",
        surface: "var(--bg-surface)",
        raised: "var(--bg-raised)",
        hair: "var(--border-hair)",
        strong: "var(--border-strong)",
        ink: {
          DEFAULT: "var(--ink-primary)",
          secondary: "var(--ink-secondary)",
          muted: "var(--ink-muted)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          ink: "var(--accent-ink)",
          dim: "var(--accent-dim)",
        },
        e3g: "var(--s-e3g)",
        pdg: "var(--s-pdg)",
        lh: "var(--s-lh)",
        calendar: "var(--calendar)",
        truth: "var(--truth)",
        good: "var(--good)",
        critical: "var(--critical)",
        warning: "var(--warning)",
      },
      fontFamily: { sans: ["var(--font-sans)"], mono: ["var(--font-mono)"] },
      borderRadius: { pill: "999px", card: "20px" },
      maxWidth: { shell: "1200px" },
    },
  },
  plugins: [],
};
export default config;
