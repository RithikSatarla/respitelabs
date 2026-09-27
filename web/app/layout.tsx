import type { Metadata } from "next";
import { Schibsted_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const grotesk = Schibsted_Grotesk({
  subsets: ["latin"],
  variable: "--font-grotesk",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono-face",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Kintrace",
  description:
    "Kintrace finds what moved on a robot arm, fixes it, and tells you it is good to go.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${grotesk.variable} ${mono.variable}`}>
      <body className="font-sans">{children}</body>
    </html>
  );
}
