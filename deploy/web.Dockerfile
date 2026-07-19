# Static export of the Next.js frontend, served by a tiny static server.
#
# NEXT_PUBLIC_API_URL has to be present at BUILD time, not run time: the app is a static export, so
# the value is inlined into the bundle rather than read from the environment on boot.

FROM node:22-alpine AS build

ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

WORKDIR /app
COPY web/package.json web/package-lock.json ./
RUN npm ci

COPY web/ ./
RUN npm run build

FROM node:22-alpine
WORKDIR /app
RUN npm install -g serve@14
COPY --from=build /app/out ./out
EXPOSE 3000
CMD ["serve", "out", "-l", "3000"]
