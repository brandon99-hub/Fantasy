import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { NotificationProvider } from "@/components/ui/NotificationProvider";
import { ReactQueryProvider } from "@/components/providers/ReactQueryProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "FPL AI Optimizer",
  description: "Advanced Fantasy Premier League optimization with AI and mathematical optimization",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} antialiased`}>
        <ReactQueryProvider>
          <NotificationProvider>{children}</NotificationProvider>
        </ReactQueryProvider>
      </body>
    </html>
  );
}