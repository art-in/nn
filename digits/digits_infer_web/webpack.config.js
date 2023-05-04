const path = require("path");
const CopyPlugin = require("copy-webpack-plugin");
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

const packDir = path.resolve(__dirname, "pack");
const buildDir = path.resolve(__dirname, "build");
const jsDir = path.resolve(__dirname, "js");
const staticDir = path.resolve(__dirname, "static");

module.exports = {
  entry: {
    index: path.resolve(jsDir, "index.js")
  },
  output: {
    path: packDir,
    filename: "[name].js"
  },
  devServer: {
    // do not log into browser console on each successful hot reload
    client: { logging: 'warn' }
  },
  plugins: [
    new CopyPlugin({ patterns: [{ from: staticDir }] }),

    // watches rust crate dir and runs wasm-pack build on changes
    new WasmPackPlugin({
      crateDirectory: __dirname,
      outDir: buildDir
    })
  ],
  experiments: {
    // allow to import wasm modules without dynamic import(), per spec proposal
    // https://github.com/WebAssembly/esm-integration. static import desugarized
    // into normal async fetch/instantiate process of wasm module (as before)
    asyncWebAssembly: true
  },
  watchOptions: {
    // enable polling to fix issue when dev server doesn't notice rebuilds
    // TODO: remove when fixed https://github.com/wasm-tool/wasm-pack-plugin/issues/125
    poll: 200,
    aggregateTimeout: 200,
    ignored: /node_modules/
  },
};