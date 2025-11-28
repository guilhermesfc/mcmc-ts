import { metropolisHastings, simpleESS, rhatAll } from "../src/index.js";
import * as fs from "fs";

const dim = 1;

// Standard normal log density (up to a constant)
const logDensity = (x: number[]) => {
  const v = x[0];
  return -0.5 * v * v; // no need to add -0.5*log(2π)
};

// Run 4 chains to assess convergence
const res = metropolisHastings(logDensity, dim, {
  chains: 4,
  iterations: 10_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
});

// Combine all chains
const allSamples = res.samples.flat();
const xs = allSamples.map((row) => row[0]);
const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

console.log("Chains:", res.samples.length);
console.log("Samples per chain:", res.samples[0].length);
console.log("Total samples:", xs.length);
console.log(
  "Acceptance rates:",
  res.acceptanceRates.map((r) => r.toFixed(3)).join(", "),
);
console.log("Sample mean (should be ~0):", mean.toFixed(3));
console.log("ESS (combined):", Math.round(simpleESS(xs)));

// R-hat convergence diagnostic
const rhats = rhatAll(res.samples);
console.log("R-hat:", rhats[0].toFixed(4));

// Create histogram data
const bins = 50;
const min = Math.min(...xs);
const max = Math.max(...xs);
const binWidth = (max - min) / bins;
const histogram = new Array(bins).fill(0);

for (const x of xs) {
  const binIndex = Math.min(Math.floor((x - min) / binWidth), bins - 1);
  histogram[binIndex]++;
}

// Create ASCII histogram
console.log("\nDistribution (histogram):");
const maxCount = Math.max(...histogram);
const height = 20;

for (let i = height; i > 0; i--) {
  const threshold = (i / height) * maxCount;
  let line = "";
  for (let j = 0; j < bins; j++) {
    line += histogram[j] >= threshold ? "█" : " ";
  }
  console.log(line);
}

// X-axis
const xAxisMin = min.toFixed(1);
const xAxisMax = max.toFixed(1);
console.log("└" + "─".repeat(bins - 2) + "┘");
console.log(
  xAxisMin + " ".repeat(bins - xAxisMin.length - xAxisMax.length) + xAxisMax,
);

// Prepare data for all chains
const chainColors = [
  "rgba(255, 99, 71, 0.7)",
  "rgba(100, 150, 200, 0.7)",
  "rgba(144, 238, 144, 0.7)",
  "rgba(255, 165, 0, 0.7)",
];

const chainData = res.samples.map((chain, i) => ({
  chain: i + 1,
  samples: chain.map((row) => row[0]),
  color: chainColors[i],
}));

// Create HTML plot
const html = `<!DOCTYPE html>
<html>
<head>
  <title>MCMC Chain Distribution (4 Chains)</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body { font-family: Arial, sans-serif; max-width: 900px; margin: 20px auto; }
    .convergence-good { color: green; }
    .convergence-ok { color: orange; }
    .convergence-bad { color: red; }
  </style>
</head>
<body>
  <h1>MCMC Sampling: 4 Chains</h1>

  <div id="histogram" style="width:100%;height:400px;"></div>
  <div id="trace" style="width:100%;height:400px;"></div>
  <div id="autocorr" style="width:100%;height:300px;"></div>

  <h3>Convergence Diagnostics</h3>
  <p>R-hat: <span id="rhat-value"></span> <span id="rhat-status"></span></p>
  <p>Acceptance rates: ${res.acceptanceRates.map((r, i) => `Chain ${i + 1}: ${r.toFixed(3)}`).join(", ")}</p>
  <p>Total samples: ${xs.length} (${res.samples[0].length} per chain)</p>

  <script>
    const chains = ${JSON.stringify(chainData)};
    const rhat = ${rhats[0]};

    // Display R-hat with color coding
    const rhatValueEl = document.getElementById('rhat-value');
    const rhatStatusEl = document.getElementById('rhat-status');
    rhatValueEl.textContent = rhat.toFixed(4);

    if (rhat < 1.01) {
      rhatStatusEl.textContent = '✓ Excellent convergence';
      rhatStatusEl.className = 'convergence-good';
    } else if (rhat < 1.05) {
      rhatStatusEl.textContent = '✓ Good convergence';
      rhatStatusEl.className = 'convergence-ok';
    } else {
      rhatStatusEl.textContent = '⚠ Chains may not have converged';
      rhatStatusEl.className = 'convergence-bad';
    }

    // Combine all samples for histogram
    const allSamples = chains.flatMap(c => c.samples);

    // Histogram with density normalization
    const histTrace = {
      x: allSamples,
      type: 'histogram',
      nbinsx: 50,
      name: 'MCMC samples (all chains)',
      histnorm: 'probability density',
      marker: { color: 'rgba(100, 150, 200, 0.7)' }
    };

    // Standard normal PDF overlay
    const xRange = [];
    const normalPdf = [];
    for (let x = -4; x <= 4; x += 0.05) {
      xRange.push(x);
      normalPdf.push(Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI));
    }

    const normalTrace = {
      x: xRange,
      y: normalPdf,
      type: 'scatter',
      mode: 'lines',
      name: 'Standard Normal PDF',
      line: { color: 'red', width: 3 }
    };

    Plotly.newPlot('histogram', [histTrace, normalTrace], {
      title: 'MCMC Sample Distribution vs Standard Normal (All Chains Combined)',
      xaxis: { title: 'Value' },
      yaxis: { title: 'Density' }
    });

    // Trace plot - all chains overlaid
    const traceData = chains.map(chainInfo => ({
      y: chainInfo.samples,
      type: 'scatter',
      mode: 'lines',
      line: { color: chainInfo.color, width: 1 },
      name: \`Chain \${chainInfo.chain}\`
    }));

    Plotly.newPlot('trace', traceData, {
      title: 'MCMC Trace Plot (All Chains)',
      xaxis: { title: 'Iteration' },
      yaxis: { title: 'Value' }
    });

    // Autocorrelation for combined samples
    const mean = allSamples.reduce((a, b) => a + b, 0) / allSamples.length;
    const variance = allSamples.reduce((s, v) => s + (v - mean) ** 2, 0) / (allSamples.length - 1);

    const maxLag = Math.min(100, allSamples.length - 1);
    const lags = [];
    const acf = [];

    for (let lag = 0; lag <= maxLag; lag++) {
      let c = 0;
      for (let i = 0; i < allSamples.length - lag; i++) {
        c += (allSamples[i] - mean) * (allSamples[i + lag] - mean);
      }
      lags.push(lag);
      acf.push(c / ((allSamples.length - 1) * variance));
    }

    const acfTrace = {
      x: lags,
      y: acf,
      type: 'bar',
      marker: { color: 'rgba(100, 150, 200, 0.7)' },
      name: 'Autocorrelation'
    };

    Plotly.newPlot('autocorr', [acfTrace], {
      title: 'Autocorrelation Function (Combined Chains)',
      xaxis: { title: 'Lag' },
      yaxis: { title: 'ACF' }
    });
  </script>
</body>
</html>`;

fs.writeFileSync("mcmc-plot.html", html);
console.log("\n✓ Interactive plot saved to mcmc-plot.html");
console.log("  Open it in your browser to see the full visualization!");
