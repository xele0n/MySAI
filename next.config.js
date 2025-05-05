/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async redirects() {
    return [
      {
        source: '/ai-playground',
        destination: '/ai-playground/',
        permanent: true,
      },
      {
        source: '/your-ai',
        destination: '/your-ai/',
        permanent: true,
      },
    ]
  },
}

module.exports = nextConfig 