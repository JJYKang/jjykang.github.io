'use strict';
const dog = window.dogData;
for (const [attribute, value] of [['size', dog.size], ['width', dog.size + 2],
  ['sigma', dog.sigma], ['threshold', dog.threshold]]) {
  document.querySelectorAll(`[data-dog-${attribute}]`).forEach(element => {
    element.textContent = value;
  });
}
document.querySelectorAll('[data-dog-error]').forEach(element => {
  const key = element.dataset.dogError;
  element.textContent = key === 'different_edge_pixels' ? dog.errors[key] : dog.errors[key].toExponential(2);
});
document.querySelectorAll('[data-edge-threshold]').forEach(element => {
  element.textContent = window.finiteDifferenceData.threshold;
});
const { results, snippets } = window.convolutionData;
document.querySelectorAll('[data-source]').forEach(element => {
  element.textContent = snippets[element.dataset.source];
});
document.getElementById('benchmark-context').textContent =
  `Each method was timed ${results.repeats} times after a warm-up run. The table reports the median runtime for the ${results.shape[1]}×${results.shape[0]} selfie.`;
const names = { box: '9×9 box', dx: 'Dx (1×3)', dy: 'Dy (3×1)' };
for (const record of results.records) {
  for (const measurement of record.measurements) {
    const row = document.createElement('tr');
    for (const value of [names[record.filter], measurement.implementation,
      measurement.milliseconds.toFixed(3), measurement.max_error.toExponential(2)]) {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    }
    document.getElementById('benchmark-body').append(row);
  }
}
const box = results.records.find(record => record.filter === 'box').measurements;
const error = Math.max(...results.records.flatMap(record => record.measurements.map(item => item.max_error)));
document.getElementById('runtime-summary').textContent =
  `The largest difference from SciPy is ${error.toExponential(2)}, so the implementations agree apart from rounding. On the box filter, the two-loop version was ${(box[0].milliseconds / box[1].milliseconds).toFixed(1)} times faster than four loops, and SciPy was another ${(box[1].milliseconds / box[2].milliseconds).toFixed(1)} times faster.`;
document.getElementById('input-caption').textContent = `${results.shape[1]}×${results.shape[0]} grayscale input.`;
