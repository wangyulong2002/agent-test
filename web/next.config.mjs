/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export", // 静态导出：out/ → 由 FastAPI 托管
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
