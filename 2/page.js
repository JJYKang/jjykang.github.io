'use strict';
document.querySelectorAll('[data-edge-threshold]').forEach(element => {
  element.textContent = window.finiteDifferenceData.threshold;
});
const { results, snippets } = window.convolutionData;
document.querySelectorAll('[data-source]').forEach(element => {
  element.textContent = snippets[element.dataset.source];
});
document.getElementById('benchmark-context').textContent =
  `${results.shape[1]}×${results.shape[0]} grayscale input, float64; one untimed warm-up and median of ${results.repeats} runs per method/filter. NumPy ${results.numpy}; SciPy ${results.scipy}.`;
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
  `All custom outputs agree with SciPy within ${error.toExponential(2)} maximum absolute error. For the box filter, the two-loop implementation takes ${(box[0].milliseconds / box[1].milliseconds).toFixed(1)}× less time than four loops; SciPy takes ${(box[1].milliseconds / box[2].milliseconds).toFixed(1)}× less time than two loops.`;
if (!results.placeholder) {
  document.getElementById('input-note').textContent = 'These results use the supplied photograph, converted to grayscale before filtering.';
  document.getElementById('input-title').textContent = 'Grayscale photograph';
  document.getElementById('input-caption').textContent = `${results.shape[1]}×${results.shape[0]} input.`;
  document.querySelector('img').alt = 'Supplied photograph converted to grayscale';
}
