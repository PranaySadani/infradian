/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',            // fully static — the demo cannot fail on a server outage
  images: { unoptimized: true },
  trailingSlash: true,
};
module.exports = nextConfig;
