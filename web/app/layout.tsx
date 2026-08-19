import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "私厨 AI · 私人厨师",
  description: "拍下食材，AI 帮你做一桌好菜",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
