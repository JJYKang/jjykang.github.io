'use strict';
const homography = window.homographyData;
document.getElementById('homography-matrix').textContent =
  homography.homography.map(row => '[ ' + row.map(value => value.toExponential(7).padStart(15)).join('  ') + ' ]').join('\n');
document.getElementById('reprojection-rmse').textContent = homography.rmse_px.toFixed(2);
