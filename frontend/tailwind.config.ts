import type { Config } from "tailwindcss";

function oklchVar(name: string): string {
  return `oklch(var(${name}) / <alpha-value>)`;
}

const config: Config = {
  darkMode: ["class"],
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: oklchVar("--background"),
        foreground: oklchVar("--foreground"),
        card: {
          DEFAULT: oklchVar("--card"),
          foreground: oklchVar("--card-foreground"),
        },
        popover: {
          DEFAULT: oklchVar("--popover"),
          foreground: oklchVar("--popover-foreground"),
        },
        primary: {
          DEFAULT: oklchVar("--primary"),
          foreground: oklchVar("--primary-foreground"),
        },
        secondary: {
          DEFAULT: oklchVar("--secondary"),
          foreground: oklchVar("--secondary-foreground"),
        },
        muted: {
          DEFAULT: oklchVar("--muted"),
          foreground: oklchVar("--muted-foreground"),
        },
        accent: {
          DEFAULT: oklchVar("--accent"),
          foreground: oklchVar("--accent-foreground"),
        },
        destructive: {
          DEFAULT: oklchVar("--destructive"),
          foreground: oklchVar("--destructive-foreground"),
        },
        border: oklchVar("--border"),
        input: oklchVar("--input"),
        ring: oklchVar("--ring"),
        chart: {
          "1": oklchVar("--chart-1"),
          "2": oklchVar("--chart-2"),
          "3": oklchVar("--chart-3"),
          "4": oklchVar("--chart-4"),
          "5": oklchVar("--chart-5"),
        },
        sidebar: {
          DEFAULT: oklchVar("--sidebar"),
          foreground: oklchVar("--sidebar-foreground"),
          primary: oklchVar("--sidebar-primary"),
          "primary-foreground": oklchVar("--sidebar-primary-foreground"),
          accent: oklchVar("--sidebar-accent"),
          "accent-foreground": oklchVar("--sidebar-accent-foreground"),
          border: oklchVar("--sidebar-border"),
          ring: oklchVar("--sidebar-ring"),
        },
      },
      borderRadius: {
        sm: "calc(var(--radius) - 4px)",
        md: "calc(var(--radius) - 2px)",
        lg: "var(--radius)",
        xl: "calc(var(--radius) + 4px)",
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
      },
    },
  },
  plugins: [],
};
export default config;
