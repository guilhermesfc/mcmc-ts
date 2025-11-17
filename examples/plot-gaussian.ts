import { metropolisHastings, simpleESS } from "../src/index.js";
import * as fs from "fs";

const dim = 1;

// Standard normal log density (up to a constant)
const logDensity = (x: number[]) => {
  const v = x[0];
  return -0.5 * v * v; // no need to add -0.5*log(2π)
};

const res = metropolisHastings(logDensity, dim, {
  iterations: 10_000,
  stepSize: 0.7,
  burnIn: 500,
  thin: 5,
  start: [5],
});

const xs = res.chain.map((row) => row[0]);
const mean = xs.reduce((a, b) => a + b, 0) / xs.length;

console.log("Samples kept:", xs.length);
console.log("Acceptance rate:", res.acceptanceRate.toFixed(3));
console.log("Sample mean (should be ~0):", mean.toFixed(3));
console.log("ESS (rough):", Math.round(simpleESS(xs)));

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
console.log(xAxisMin + " ".repeat(bins - xAxisMin.length - xAxisMax.length) + xAxisMax);

// Create HTML plot
const html = `<!DOCTYPE html>
<html>
<head>
  <title>MCMC Chain Distribution</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
</head>
<body>
  <div id="histogram" style="width:800px;height:400px;"></div>
  <div id="trace" style="width:800px;height:300px;"></div>
  <div id="autocorr" style="width:800px;height:300px;"></div>

  <script>
    const samples = ${JSON.stringify(xs)};

    // Histogram with density normalization
    const histTrace = {
      x: samples,
      type: 'histogram',
      nbinsx: 50,
      name: 'MCMC samples',
      histnorm: 'probability density',
      marker: { color: 'rgba(100, 150, 200, 0.7)' }
    };

    // Standard normal PDF overlay (properly scaled)
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
      title: 'MCMC Sample Distribution vs Standard Normal (Density)',
      xaxis: { title: 'Value' },
      yaxis: { title: 'Density' }
    });

    // Trace plot
    const traceData = {
      y: samples,
      type: 'scatter',
      mode: 'lines',
      line: { color: 'rgba(100, 150, 200, 0.5)', width: 1 },
      name: 'Chain values'
    };

    Plotly.newPlot('trace', [traceData], {
      title: 'MCMC Trace Plot',
      xaxis: { title: 'Iteration' },
      yaxis: { title: 'Value' }
    });

    // Autocorrelation
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((s, v) => s + (v - mean) ** 2, 0) / (samples.length - 1);

    const maxLag = Math.min(100, samples.length - 1);
    const lags = [];
    const acf = [];

    for (let lag = 0; lag <= maxLag; lag++) {
      let c = 0;
      for (let i = 0; i < samples.length - lag; i++) {
        c += (samples[i] - mean) * (samples[i + lag] - mean);
      }
      lags.push(lag);
      acf.push(c / ((samples.length - 1) * variance));
    }

    const acfTrace = {
      x: lags,
      y: acf,
      type: 'bar',
      marker: { color: 'rgba(100, 150, 200, 0.7)' },
      name: 'Autocorrelation'
    };

    Plotly.newPlot('autocorr', [acfTrace], {
      title: 'Autocorrelation Function',
      xaxis: { title: 'Lag' },
      yaxis: { title: 'ACF' }
    });
  </script>
</body>
</html>`;

fs.writeFileSync("mcmc-plot.html", html);
console.log("\n✓ Interactive plot saved to mcmc-plot.html");
console.log("  Open it in your browser to see the full visualization!");
