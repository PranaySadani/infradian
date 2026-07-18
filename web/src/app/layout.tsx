import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

// Closest freely-licensable match to MatterSQ: geometric sans, medium-weight display
// character, holds tight negative tracking at large sizes. Self-hosted by next/font,
// so the static export has no runtime CDN dependency.
const jakarta = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-jakarta",
  display: "swap",
});

export const metadata: Metadata = {
  title: "INFRADIAN — the rhythm nobody benchmarks",
  description:
    "An open benchmark for hormonal trajectory inference from consumer wearables. Skill over Calendar.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={jakarta.variable}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
