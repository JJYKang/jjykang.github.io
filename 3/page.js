'use strict';
const homography = window.homographyData;
document.getElementById('homography-matrix').textContent =
  homography.homography.map(row => '[ ' + row.map(value => value.toExponential(7).padStart(15)).join('  ') + ' ]').join('\n');
document.getElementById('reprojection-rmse').textContent = homography.rmse_px.toFixed(2);

const warping = window.warpingData;
document.getElementById('warp-repeats').textContent = warping.repeats;
for (const record of warping.records) {
  const row = document.createElement('tr');
  const name = record.name === 'sign' ? 'Clinic sign' : 'House window';
  for (const value of [name, `${record.output_shape[1]}×${record.output_shape[0]}`,
    record.milliseconds.nearest.toFixed(2), record.milliseconds.bilinear.toFixed(2)]) {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.append(cell);
  }
  document.getElementById('warp-timings').append(row);
}
