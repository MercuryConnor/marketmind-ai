import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./src/**/*.{ts,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "var(--mm-bg)",
        surface: "var(--mm-surface)",
        panel: "var(--mm-panel)",
        border: "var(--mm-border)",
        text: "var(--mm-text)",
        muted: "var(--mm-muted)",
        accent: "var(--mm-accent)",
        accentSoft: "var(--mm-accent-soft)",
        positive: "var(--mm-positive)",
        negative: "var(--mm-negative)",
        warning: "var(--mm-warning)",
      },
      fontFamily: {
        sans: ["var(--font-inter)", "ui-sans-serif", "system-ui"],
        mono: ["var(--font-jetbrains-mono)", "ui-monospace", "SFMono-Regular"],
      },
      borderRadius: {
        xl: "0.75rem",
        "2xl": "1rem",
      },
      boxShadow: {
        glow: "0 0 0 1px color-mix(in srgb, var(--mm-accent) 40%, transparent)",
      },
    },
  },
  plugins: [],
};

export default config;
