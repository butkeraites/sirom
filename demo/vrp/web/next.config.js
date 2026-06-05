/** @type {import('next').NextConfig} */
// The browser talks to the VRP backend directly (see NEXT_PUBLIC_VRP_API in the
// app) — a SIROM solve runs longer than a server-side proxy's request timeout.
const nextConfig = {
  output: "standalone",
};

module.exports = nextConfig;
