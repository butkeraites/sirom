# Next.js web app for the SIROM robust-CVRP demo. Multi-stage: install + build,
# then run the standalone server on a slim runtime.
FROM node:20-slim AS deps
WORKDIR /app
COPY package.json ./
RUN npm install

FROM node:20-slim AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# NEXT_PUBLIC_* is inlined at build time. This is the backend URL as seen from
# the *browser* (the host), i.e. the docker-compose published port. Local
# `npm run dev` uses the app's fallback (http://localhost:8801).
ARG NEXT_PUBLIC_VRP_API=http://localhost:8801
ENV NEXT_PUBLIC_VRP_API=$NEXT_PUBLIC_VRP_API \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

# Standalone output bundles only what the server needs.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
