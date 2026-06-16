#!/usr/bin/env node
'use strict';
// Score a FlatPPL binding's log-density at a point, using the flatppl-js engine.
// The JS-engine counterpart to the `*_root.py` oracles in this directory.
//
// Usage:
//   node score_js.cjs <model.flatppl> <binding> <theta> [--engine <dir>] [--count N]
//
//   <binding>  a measure / likelihood binding in the model (e.g. `obs`, `likelihood`).
//   <theta>    a FlatPPL record literal of the binding's free parameters, e.g.
//              'record(mu = 0.0, sigma = 1.0)'. The script appends
//              `__score__ = logdensityof(<binding>, <theta>)` to the model source
//              and prints the resulting per-atom log-density to stdout.
//
// Engine resolution (first that exists): --engine <dir>, then $FLATPPL_JS_DIR
// (its packages/engine), then a sibling ../flatppl-js checkout. Requires Node >= 24
// — the engine modules are TypeScript, loaded via Node's native type stripping.
//
// Examples (from this directory, with a sibling flatppl-js checkout):
//   node score_js.cjs gaussian.flatppl obs    'record(mu = 0.0, sigma = 1.0)'
//   node score_js.cjs product.flatppl  likelihood 'record(mu1 = 0.0, sigma1 = 1.0, mu2 = 1.0, sigma2 = 2.0)'

const fs = require('fs');
const path = require('path');

function usage(msg) {
  if (msg) process.stderr.write('score_js: ' + msg + '\n');
  process.stderr.write(
    'usage: node score_js.cjs <model.flatppl> <binding> <theta> [--engine <dir>] [--count N]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const pos = [];
  let engineDir = null;
  let count = 1;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--engine') { engineDir = argv[++i]; }
    else if (a === '--count') { count = parseInt(argv[++i], 10); }
    else if (a.startsWith('--')) { usage('unknown flag ' + a); }
    else { pos.push(a); }
  }
  if (pos.length !== 3) usage('expected <model.flatppl> <binding> <theta>');
  return { model: pos[0], binding: pos[1], theta: pos[2], engineDir, count };
}

function resolveEngine(explicit) {
  const candidates = [
    explicit,
    process.env.FLATPPL_JS_DIR && path.join(process.env.FLATPPL_JS_DIR, 'packages', 'engine'),
    path.resolve(__dirname, '..', '..', 'flatppl-js', 'packages', 'engine'),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'index.ts'))) return c;
  }
  usage('flatppl-js engine not found — pass --engine <dir> or set FLATPPL_JS_DIR '
    + '(looked in: ' + candidates.join(', ') + ')');
}

async function main() {
  const { model, binding, theta, engineDir, count } = parseArgs(process.argv.slice(2));
  const engine = resolveEngine(engineDir);
  const { processSource, orchestrator, materialiser } = require(path.join(engine, 'index.ts'));
  const { createWorkerHandler } = require(path.join(engine, 'worker.ts'));

  const base = fs.readFileSync(model, 'utf8');
  const src = base + `\n__score__ = logdensityof(${binding}, ${theta})\n`;

  const proc = processSource(src);
  // Surface diagnostics, but do NOT abort on them: a diagnostic on a binding
  // off the scored measure's dependency path (e.g. a single-factor `cartprod`
  // domain the engine flags) does not affect the density. If a diagnostic does
  // break the scored binding, materialising it below throws and is reported.
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error') process.stderr.write('diagnostic: ' + d.message + '\n');
  }

  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: count,
    rootKey: 3,
    rootSeed: 3,
    marginalizationCount: 32,
    // The standard-module registry (e.g. `hepphys`) must be threaded into the
    // worker session env, or cross-module refs in the body fail to resolve.
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m) => Promise.resolve(w.handle(m)),
  };

  const measure = await ctx.getMeasure('__score__');
  if (!measure || !measure.samples || measure.samples.length === 0) {
    process.stderr.write('score_js: no density produced for `' + binding + '`\n');
    process.exit(1);
  }
  process.stdout.write(measure.samples[0] + '\n');
}

main().catch((e) => {
  process.stderr.write('score_js: ' + (e && e.message ? e.message : e) + '\n');
  process.exit(1);
});
