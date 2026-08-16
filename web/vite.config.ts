/// <reference types="vitest" />
import fs from "fs";
import path, { resolve } from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import monacoEditorPlugin from "vite-plugin-monaco-editor";

const proxyHost = process.env.PROXY_HOST || "localhost:5001";
const liveProxyHost = process.env.LIVE_PROXY_HOST || proxyHost;
const liveProxyProtocol = process.env.LIVE_PROXY_PROTOCOL || "ws";
const mediaProxyHost = process.env.MEDIA_PROXY_HOST || "localhost:8971";
const mediaProxyProtocol = process.env.MEDIA_PROXY_PROTOCOL || "http";
const tlsCert = process.env.CAMERA_FRONTEND_TLS_CERT;
const tlsKey = process.env.CAMERA_FRONTEND_TLS_KEY;

// https://vitejs.dev/config/
export default defineConfig({
  define: {
    "import.meta.vitest": "undefined",
  },
  server: {
    https:
      tlsCert && tlsKey
        ? { cert: fs.readFileSync(tlsCert), key: fs.readFileSync(tlsKey) }
        : undefined,
    proxy: {
      "/api/runtime": {
        target: "http://frigate:5001",
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "/api": {
        target: `http://${proxyHost}`,
        ws: true,
      },
      "/vod": {
        target: `${mediaProxyProtocol}://${mediaProxyHost}`,
        secure: false,
      },
      "/clips": {
        target: `${mediaProxyProtocol}://${mediaProxyHost}`,
        secure: false,
      },
      "/exports": {
        target: `${mediaProxyProtocol}://${mediaProxyHost}`,
        secure: false,
      },
      "/recordings": {
        target: `${mediaProxyProtocol}://${mediaProxyHost}`,
        secure: false,
      },
      "/stream": {
        target: `${mediaProxyProtocol}://${mediaProxyHost}`,
        secure: false,
      },
      "/cache": {
        target: `${mediaProxyProtocol}://${mediaProxyHost}`,
        secure: false,
      },
      "/ws": {
        target: `${liveProxyProtocol}://${liveProxyHost}`,
        secure: false,
        ws: true,
      },
      "/live": {
        target: `${liveProxyProtocol}://${liveProxyHost}`,
        changeOrigin: true,
        secure: false,
        ws: true,
      },
    },
  },
  build: {
    reportCompressedSize: false,
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
        login: resolve(__dirname, "login.html"),
      },
    },
  },
  plugins: [
    react(),
    monacoEditorPlugin.default({
      customWorkers: [{ label: "yaml", entry: "monaco-yaml/yaml.worker" }],
      languageWorkers: ["editorWorkerService"], // we don't use any of the default languages
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    alias: {
      "testing-library": path.resolve(
        __dirname,
        "./__test__/testing-library.js",
      ),
    },
    setupFiles: ["./__test__/test-setup.ts"],
    includeSource: ["src/**/*.{js,jsx,ts,tsx}"],
    coverage: {
      reporter: ["text-summary", "text"],
    },
    mockReset: true,
    restoreMocks: true,
    globals: true,
  },
});
