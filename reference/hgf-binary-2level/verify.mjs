import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

const require = createRequire(import.meta.url);
const enginePath = process.env.FLATPPL_JS_DIR || fileURLToPath(new URL('../../../flatppl-js/', import.meta.url));
const { processSource, orchestrator, materialiser } = require(`${enginePath}/packages/engine/index.ts`);
const { createWorkerHandler } = require(`${enginePath}/packages/engine/worker.ts`);
const { buildLogPi } = require(`${enginePath}/packages/engine/mcmc-density.ts`);
const source = fs.readFileSync(new URL('../../examples/hgf-binary-2level.flatppl', import.meta.url), 'utf8');
const inputs = [1, 0, 1, 1, 0, 0];
const choices = [1, 0, 1, 0, 1, 0];
const points = [{ omega: -2, log_noise: 0 }, { omega: -3, log_noise: Math.log(0.7) }, { omega: -0.5, log_noise: Math.log(1.4) }];

function close(actual, expected, label) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) < 2e-12,
    `${label}: ${actual} vs ${expected}`);
}

// Independent variance-form recurrence, and closed-form Bernoulli/Normal scores.
function oracle(point, outcomes = inputs, actions = choices) {
  let mean = 0;
  let variance = 1;
  const means = [], precisions = [], probabilities = [], trialScores = [];
  for (let i = 0; i < outcomes.length; i++) {
    const predictedVariance = variance + Math.exp(point.omega);
    const predictedProbability = 1 / (1 + Math.exp(-mean));
    variance = predictedVariance / (1 + predictedVariance * predictedProbability * (1 - predictedProbability));
    mean += variance * (outcomes[i] - predictedProbability);
    const logOdds = mean / Math.exp(point.log_noise);
    means.push(mean);
    precisions.push(1 / variance);
    probabilities.push(1 / (1 + Math.exp(-logOdds)));
    trialScores.push(actions[i] * logOdds - Math.log1p(Math.exp(logOdds)));
  }
  const likelihood = trialScores.reduce((a, b) => a + b, 0);
  const normalLogpdf = (x, mu, sd) => -0.5 * Math.log(2 * Math.PI) - Math.log(sd) - 0.5 * ((x - mu) / sd) ** 2;
  return { means, precisions, probabilities, trialScores, likelihood,
    posterior: likelihood + normalLogpdf(point.omega, -2, 1) + normalLogpdf(point.log_noise, 0, 0.5) };
}

function makeContext(modelSource, sampleCount = 1) {
  const lifted = processSource(modelSource);
  const built = orchestrator.buildDerivations(lifted.linkedBindings);
  const errors = [...lifted.diagnostics || [], ...built.diagnostics || []].filter(d => d.severity === 'error');
  assert.deepEqual(errors, []);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 123 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount, rootSeed: 123,
    getMeasure(name) {
      if (!cache.has(name)) cache.set(name, materialiser.materialiseMeasure(name, ctx));
      return cache.get(name);
    },
    sendWorker(msg) {
      const reply = worker.handle(msg);
      if (reply?.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
  };
  return { ctx, built };
}

async function score(point, modelSource = source) {
  const pointSource = `record(omega = ${point.omega}, log_noise = ${point.log_noise})`;
  const { ctx } = makeContext(`${modelSource}\n
checked_means = mean_trajectory(omega = ${point.omega})
checked_precisions = precision_trajectory(omega = ${point.omega})
checked_probabilities = choice_probabilities(${pointSource})
checked_likelihood = logdensityof(L, ${pointSource})
checked_posterior = logdensityof(posterior, ${pointSource})
`);
  const values = {};
  for (const key of ['means', 'precisions', 'probabilities', 'likelihood', 'posterior']) {
    const result = await ctx.getMeasure(`checked_${key}`);
    assert.ok(result?.samples, `missing ${key}`);
    values[key] = ['likelihood', 'posterior'].includes(key) ? result.samples[0] : Array.from(result.samples);
  }
  return values;
}

function compare(actual, expected) {
  for (const key of ['means', 'precisions', 'probabilities']) {
    assert.equal(actual[key].length, expected[key].length);
    actual[key].forEach((value, i) => close(value, expected[key][i], `${key}[${i}]`));
  }
  close(actual.likelihood, expected.likelihood, 'likelihood');
  close(actual.posterior, expected.posterior, 'posterior');
}

test('FlatPPL HGF states and scores match closed-form maths', async () => {
  for (const point of points) compare(await score(point), oracle(point));
});

test('outcome order drives learning, and choices only score the learned beliefs', async () => {
  const reversed = inputs.toReversed();
  const reorderedSource = source.replace('u_data = [1, 0, 1, 1, 0, 0]', `u_data = [${reversed}]`);
  compare(await score(points[0], reorderedSource), oracle(points[0], reversed));
  const flipped = choices.map(x => 1 - x);
  const flippedSource = source.replace('choice_data = [1, 0, 1, 0, 1, 0]', `choice_data = [${flipped}]`);
  compare(await score(points[0], flippedSource), oracle(points[0], inputs, flipped));
});

test('HGF posterior sampling scores its actual draws against independent maths', async () => {
  const { ctx, built } = makeContext(source, 128);
  const result = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 1, warmup: 100, draws: 128, seed: 1,
  });
  const scorer = await buildLogPi(ctx, built.derivations.posterior);
  const omega = result.fields.omega.samples;
  const logNoise = result.fields.log_noise.samples;
  assert.ok(new Set(omega).size > 1, 'the chain must move');
  for (const i of [0, 64, 127]) {
    const point = { omega: omega[i], log_noise: logNoise[i] };
    const expected = oracle(point);
    close(scorer.likOf(point), expected.likelihood, 'sample likelihood');
    close(scorer.logPi(point), expected.posterior, 'sample posterior');
  }
});

const referencePath = process.argv[2] || new URL('julia-reference.json', import.meta.url);
test('FlatPPL HGF states and scores match the original Julia package', async () => {
  const reference = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
  assert.deepEqual(reference.inputs, inputs);
  assert.deepEqual(reference.choices, choices);
  assert.equal(reference.cases.length, points.length);
  for (let i = 0; i < points.length; i++) {
    const row = reference.cases[i];
    close(row.omega, points[i].omega, 'omega');
    close(row.log_noise, points[i].log_noise, 'log_noise');
    compare(row, oracle(points[i]));
    compare(await score(points[i]), row);
  }
});
