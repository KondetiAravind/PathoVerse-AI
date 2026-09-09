import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PathoVerse AI",
  description:
    "Multimodal Foundation Model Platform for Whole-Slide Pathology Analysis & Evaluation",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}