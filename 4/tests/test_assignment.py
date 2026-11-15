"""Independent geometry checks and report/result consistency for Assignment 4."""
from html.parser import HTMLParser
import importlib
import json
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / ".deps"))

import numpy as np
from PIL import Image
import fundamental as fund
import rectification as rect
import factorization as fact
import triangulation as tri
from warping import warpImageBilinear


def skew(t):
    x, y, z = t
    return np.array([[0., -z, y], [z, 0, -x], [-y, x, 0]])


def canonical(matrix):
    matrix = matrix / np.linalg.norm(matrix)
    return matrix if matrix.flat[np.argmax(abs(matrix))] > 0 else -matrix


def project(points, cameras):
    values = np.einsum('mij,nj->nmi', cameras, np.column_stack((points, np.ones(len(points)))))
    return values[:, :, :2] / values[:, :, 2:3]


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(671)
        self.k = np.array([[800., 0, 320], [0, 790, 240], [0, 0, 1]])
        a = .2
        self.r = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        self.t = np.array([.9, .1, .2])
        self.t /= np.linalg.norm(self.t)
        self.e = skew(self.t) @ self.r
        self.f = np.linalg.inv(self.k).T @ self.e @ np.linalg.inv(self.k)
        self.cameras = np.stack((self.k @ np.column_stack((np.eye(3), np.zeros(3))),
                                 self.k @ np.column_stack((self.r, self.t))))
        self.world = self.rng.uniform([-1, -1, 4], [1, 1, 8], (30, 3))
        self.observations = project(self.world, self.cameras)
        self.p = np.column_stack((self.observations[:, 0], np.ones(30)))
        self.q = np.column_stack((self.observations[:, 1], np.ones(30)))

    def test_eight_point_methods_recover_known_geometry(self):
        for count in (8, 30):
            for method in (fund.lls_eight_point_alg, fund.normalized_eight_point_alg):
                f = method(self.p[:count], self.q[:count])
                np.testing.assert_allclose(f, canonical(self.f), atol=1e-8)
                self.assertEqual(np.linalg.matrix_rank(f), 2)

    def test_normalization_and_coordinate_equivariance(self):
        points, transform = fund._normalize_points(self.p)
        np.testing.assert_allclose(points[:, :2].mean(0), 0, atol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1).mean(), np.sqrt(2))
        np.testing.assert_allclose(points, self.p @ transform.T, atol=1e-13)
        noisy = self.q.copy()
        noisy[:, :2] += self.rng.normal(0, .2, (30, 2))
        f = fund.normalized_eight_point_alg(self.p, noisy)
        t1 = np.array([[3., 0, 1000], [0, 3, -900], [0, 0, 1]])
        t2 = np.array([[.4, 0, -200], [0, .4, 700], [0, 0, 1]])
        changed = fund.normalized_eight_point_alg(self.p @ t1.T, noisy @ t2.T)
        np.testing.assert_allclose(canonical(t2.T @ changed @ t1), f, atol=1e-9)

    def test_distance_direction_and_arbitrary_scale(self):
        p = np.array([[1., 5, 1], [4, 7, 1]])
        q = np.array([[2., 9, 1], [-1, 3, 1]])
        f = np.array([[0., 0, 1], [0, 0, 0], [2, 0, 3]])
        self.assertAlmostEqual(fund.compute_distance_to_epipolar_lines(p, q, f), 4.25)
        self.assertAlmostEqual(fund.compute_distance_to_epipolar_lines(q, p, f.T), 8.5)
        for scale in (-7., 1e-180, 1e180):
            self.assertAlmostEqual(fund.compute_distance_to_epipolar_lines(p*3, q*2, f*scale), 4.25)

    def test_epipoles_and_mapping_to_infinity(self):
        expected = self.k @ self.t
        np.testing.assert_allclose(rect.compute_epipole(self.f), expected/expected[2], atol=1e-8)
        image = np.zeros((480, 640))
        for e in (expected, np.array([-500., 100, 1]), np.array([320., 900, 1]),
                  np.array([1., 0, 0]), np.array([0., 1, 0])):
            h = rect.compute_H(e, image)
            mapped = h @ e
            np.testing.assert_allclose(mapped[1:]/abs(mapped[0]), 0, atol=1e-12)
            center = h @ [320, 240, 1]
            np.testing.assert_allclose(center/center[2], [320, 240, 1], atol=1e-10)
            np.testing.assert_allclose(rect.compute_H(-3*e, image), h, atol=1e-10)

    def test_matching_homographies_preserve_scanlines(self):
        h1, h2 = rect.compute_matching_homographies(rect.compute_epipole(self.f), self.f,
                                                   np.zeros((480, 640)), self.p, self.q)
        p = self.p @ h1.T
        q = self.q @ h2.T
        np.testing.assert_allclose(p[:, 1]/p[:, 2], q[:, 1]/q[:, 2], atol=1e-8)
        transformed = np.linalg.inv(h2).T @ self.f @ np.linalg.inv(h1)
        np.testing.assert_allclose(canonical(transformed), canonical(skew([1., 0, 0])), atol=1e-9)

    def test_bilinear_warp_identity_and_fractional_ramp(self):
        ramp = np.arange(30.).reshape(5, 6)
        warped, mask, _ = warpImageBilinear(ramp, np.eye(3))
        np.testing.assert_array_equal(warped, ramp)
        self.assertTrue(mask.all())
        h = np.array([[1., 0, .25], [0, 1, .5], [0, 0, 1]])
        warped, mask, _ = warpImageBilinear(ramp, h, (3, 4), (1, 1))
        y, x = np.mgrid[1:4, 1:5]
        np.testing.assert_allclose(warped, (y-.5)*6+x-.25)

    def test_factorization_exact_affine_and_noisy_optimum(self):
        for noise in (0, .1):
            world = self.rng.normal(size=(3, 25))
            measurements = self.rng.normal(size=(4, 3)) @ world + [[200], [100], [350], [-75]]
            measurements += self.rng.normal(0, noise, measurements.shape)
            p = np.column_stack((measurements[:2].T, np.ones(25)))
            q = np.column_stack((measurements[2:].T, np.ones(25)))
            structure, motion = fact.factorization_method(p, q)
            centered, translation = fact.build_measurement_matrix(p, q)
            self.assertEqual(structure.shape, (3, 25))
            self.assertEqual(motion.shape, (4, 3))
            np.testing.assert_allclose(structure.mean(1), 0, atol=1e-12)
            discarded = np.linalg.svd(centered, compute_uv=False)[3]
            np.testing.assert_allclose(np.linalg.norm(centered-motion @ structure), discarded, atol=1e-12)
            if noise == 0:
                np.testing.assert_allclose(motion @ structure+translation, measurements, atol=1e-12)

    def test_four_pose_candidates_and_positive_depth_selection(self):
        expected = np.column_stack((self.r, self.t))
        candidates = tri.estimate_initial_RT(self.e)
        self.assertEqual(candidates.shape, (4, 3, 4))
        self.assertEqual(len({tuple(np.round(p.ravel(), 10)) for p in candidates}), 4)
        for candidate in candidates:
            r, t = candidate[:, :3], candidate[:, 3]
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r), 1)
            self.assertAlmostEqual(np.linalg.norm(t), 1)
            np.testing.assert_allclose(canonical(skew(t) @ r), canonical(self.e), atol=1e-12)
        for observations in (self.observations, self.observations[:1]):
            for sign in (1, -1):
                np.testing.assert_allclose(tri.estimate_RT_from_E(sign*self.e, observations, self.k), expected, atol=1e-10)

    def test_linear_triangulation_exact_and_infinite_point(self):
        for world in self.world[:5]:
            points = project(world[None], self.cameras)[0]
            np.testing.assert_allclose(tri.linear_estimate_3d_point(points, self.cameras), world, atol=1e-12)
        cameras = np.stack((np.column_stack((np.eye(3), [0., 0, 0])),
                            np.column_stack((np.eye(3), [1., 0, 0]))))
        with self.assertRaises(ValueError):
            tri.linear_estimate_3d_point(np.zeros((2, 2)), cameras)

    def test_residual_order_and_analytic_derivatives(self):
        camera = np.array([[[2., 0, 0, 3], [0, 4, 0, -2], [0, 0, 1, 1]]])
        np.testing.assert_allclose(tri.reprojection_error([1, 2, 3], [[.25, -.5]], camera), [1, 2])
        np.testing.assert_allclose(tri.jacobian([1, 2, 3], camera), [[.5, 0, -.3125], [0, 1, -.375]])
        point = self.world[0]
        observed = self.observations[0]
        j = tri.jacobian(point, self.cameras)
        finite = np.column_stack([(tri.reprojection_error(point+np.eye(3)[i]*1e-5, observed, self.cameras)
                                   - tri.reprojection_error(point-np.eye(3)[i]*1e-5, observed, self.cameras))/(2e-5)
                                  for i in range(3)])
        np.testing.assert_allclose(j, finite, atol=1e-8, rtol=1e-7)

    def test_gauss_newton_ten_updates_and_independent_optimum(self):
        cameras = np.array([[[1., 0, 0, .5], [0, 1, 0, -.2], [0, 0, 0, 1]],
                            [[1., 0, 1, -.3], [0, 1, .4, .8], [0, 0, 0, 1]]])
        observed = project(self.world[:1], cameras)[0] + [[.1, -.1], [.2, .05]]
        expected = np.linalg.lstsq(cameras[:, :2, :3].reshape(-1, 3),
                                   (observed-cameras[:, :2, 3]).ravel(), rcond=None)[0]
        with patch.object(tri, 'jacobian', wraps=tri.jacobian) as calls:
            result = tri.nonlinear_estimate_3d_point(observed, cameras)
            self.assertEqual(calls.call_count, 10)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_dense_batch_matches_scalar_and_parallax_filter(self):
        from sfm_utils import triangulate_dense_matches, remove_outliers
        from types import SimpleNamespace
        points, valid = triangulate_dense_matches(self.observations[:8], self.cameras, batch_size=3)
        self.assertTrue(valid.all())
        reference = np.array([tri.nonlinear_estimate_3d_point(p, self.cameras) for p in self.observations[:8]])
        np.testing.assert_allclose(points, reference, atol=1e-10)
        poses = np.array([np.column_stack((np.eye(3), [-i, 0., 0.])) for i in range(3)])
        points = np.array([[0., 0, 4], [0, 0, 1000.]])
        observations = project(points, poses).transpose(1, 0, 2).reshape(-1, 2)
        frame = SimpleNamespace(K=np.eye(3), motion=poses, structure=points.copy(),
                                match_idx=np.arange(6).reshape(3, 2), match_points=observations)
        remove_outliers(frame)
        np.testing.assert_allclose(frame.structure, points[:1])

    def test_required_entrypoint_apis(self):
        expected = {1: ['lls_eight_point_alg', 'normalized_eight_point_alg', 'compute_distance_to_epipolar_lines'],
                    2: ['compute_epipole', 'compute_H', 'compute_matching_homographies'],
                    3: ['factorization_method'],
                    4: ['estimate_initial_RT', 'linear_estimate_3d_point', 'reprojection_error',
                        'jacobian', 'nonlinear_estimate_3d_point', 'estimate_RT_from_E']}
        for part, names in expected.items():
            module = importlib.import_module(f'p{part}')
            for name in names:
                self.assertTrue(callable(getattr(module, name)))


class ReportTests(unittest.TestCase):
    def test_every_pdf_subpart_has_an_answer(self):
        page = (ROOT / 'index.html').read_text(encoding='utf-8-sig')
        for number, letters in [(1, 'abcdef'), (2, 'abcd'), (3, 'abcdef'), (4, 'abcdef')]:
            for letter in letters:
                section = re.search(rf'<h3 id="part{number}{letter}">.*?</h3>(.*?)(?=<h3|</section>)', page, re.S)
                self.assertIsNotNone(section)
                self.assertTrue(re.sub('<[^>]+>', '', section[1]).strip())
        self.assertNotIn('Not computed', page)
        self.assertNotIn('Pending', page)

    def test_local_links_images_and_declared_dimensions(self):
        class Assets(HTMLParser):
            def handle_starttag(inner, tag, attrs):
                a = dict(attrs)
                if tag not in ('a', 'img', 'link', 'script'):
                    return
                url = urlsplit(a.get('href', a.get('src', '')))
                if url.scheme or not url.path:
                    return
                target = ROOT / unquote(url.path)
                self.assertTrue(target.is_file(), str(target))
                if tag == 'img':
                    self.assertTrue(a.get('alt'))
                    with Image.open(target) as image:
                        if 'width' in a:
                            self.assertEqual(image.size, (int(a['width']), int(a['height'])))
                        image.verify()
        Assets().feed((ROOT / 'index.html').read_text(encoding='utf-8-sig'))

    def test_report_matrices_distances_and_singular_values(self):
        page = (ROOT / 'index.html').read_text(encoding='utf-8-sig')
        records = json.loads((ROOT / 'media/part1/results.json').read_text())['records']
        blocks = re.findall(r'<pre><code>(\[\[.*?)</code></pre>', page, re.S)
        ordered = [r for method in ('linear', 'normalized') for r in records if r['method'] == method]
        self.assertEqual(len(blocks), 4)
        for text, record in zip(blocks, ordered):
            displayed = np.array([float(x) for x in re.findall(r'[+-]?\d+\.\d+e[+-]\d+', text)]).reshape(3, 3)
            np.testing.assert_allclose(displayed, record['fundamental_matrix'], rtol=5e-9)
            for key in ('mean_distance_image1_px', 'mean_distance_image2_px'):
                self.assertIn(f'{record[key]:.4f}', page)
        records = json.loads((ROOT / 'media/part3/results.json').read_text())['records']
        for record in records:
            for value in record['singular_values']:
                self.assertIn(f'{value:.4f}', page)
            comparison = np.load(ROOT / 'media/part3' / f"{record['dataset']}_comparison.npz")
            error = np.linalg.norm(comparison['display_aligned_structure']-comparison['ground_truth'], axis=1)
            self.assertAlmostEqual(np.sqrt(np.mean(error**2)), record['ground_truth_comparison']['point_rmse'])

    def test_saved_triangulation_checks_and_complete_reconstruction(self):
        records = json.loads((ROOT / 'media/part4/results.json').read_text())
        self.assertLess(records['reprojection_check']['max_finite_difference_jacobian_error'], 1e-5)
        nonlinear = records['nonlinear_triangulation_check']
        self.assertLess(nonlinear['nonlinear_reprojection_error_norm'], nonlinear['linear_reprojection_error_norm'])
        for record in records['records']:
            self.assertEqual(record['positive_depth_counts'][record['selected_candidate_index']], record['correspondences'])
        results = json.loads((ROOT / 'media/part4/reconstruction_results.json').read_text())
        arrays = np.load(ROOT / 'media/part4/statue_reconstruction.npz')
        points, cameras = arrays['dense_structure'], arrays['camera_motion']
        self.assertEqual(len(cameras), 5)
        self.assertEqual(results['dense_pair_count'], 4)
        self.assertEqual(len(points), sum(p['retained_points'] for p in results['pairs']))
        self.assertTrue(np.isfinite(points).all())
        np.testing.assert_allclose(cameras[0], np.column_stack((np.eye(3), np.zeros(3))), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(cameras[1, :, 3]), 1)
        start = 0
        for index, pair in enumerate(results['pairs']):
            current = points[start:start+pair['retained_points']]
            start += len(current)
            for camera in cameras[index:index+2]:
                self.assertTrue(((current @ camera[:, :3].T+camera[:, 3])[:, 2] > 0).all())


if __name__ == '__main__':
    unittest.main()
