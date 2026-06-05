import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SIROM — Robust CVRP Demo",
  description:
    "Solve VRP-REP vehicle routing instances under uncertainty with SIROM.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
