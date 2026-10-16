"""无需6000SKU数据的契约测试；python -m unittest discover -s tests -v。"""
import ast
from pathlib import Path
import tempfile
import unittest

import numpy as np

from replenishment.data import validate as validate_data
from replenishment.evaluation import InventoryEvaluator, metrics, simulate, target_policy
from replenishment.features import build_supervised_data
from replenishment.registry import MODELS


def fixture(n=24):
    """固定种子的合成输入，仅用于链路检查，绝不是正式实验结果。"""
    rng = np.random.default_rng(42)
    level = rng.uniform(20, 150, (n, 1))
    d = rng.poisson(level * np.linspace(.8, 1.5, 88)[None, :]).astype(float)
    x = rng.normal(size=(n, 88, 20))
    return dict(train_demand=d[:, :60], test_demand=d[:, 60:], train_x=x[:, :60],
                test_x=x[:, 60:], initial=np.full(n, 150.),
                item_ids=np.array([str(j) for j in range(n)]),
                categories=np.array(['a' if j % 2 else 'b' for j in range(n)]))


class CoreTests(unittest.TestCase):
    def test_model_count(self):
        self.assertEqual(len(MODELS), 29)
        self.assertEqual(len(set(MODELS)), 29)

    def test_data_schema(self):
        validate_data(fixture())
        raw = fixture()
        raw['train_demand'][0, 0] = -1
        with self.assertRaises(ValueError):
            validate_data(raw)

    def test_excel_input(self):
        from openpyxl import Workbook
        from replenishment.data import workbook
        with tempfile.TemporaryDirectory(prefix='yuan_excel_') as folder:
            path = Path(folder) / 'tiny.xlsx'
            book = Workbook()
            book.remove(book.active)
            for k in [1, 2]:
                sheet = book.create_sheet(f'Period_{k}')
                sheet.append(['item_id', 'category', 'initial_inventory', f'demand_{k}']
                             + [f'x{j}' for j in range(1, 21)])
                for i in range(8):
                    sheet.append([i, 'a', 100, k + i] + [float(i)] * 20)
            book.save(path)
            raw = workbook(path)
            self.assertEqual(raw['demand'].shape, (8, 2))
            self.assertEqual(raw['x'].shape, (8, 2, 20))
            np.testing.assert_array_equal(raw['demand'][0], [1, 2])

    def test_lead_one_and_sales(self):
        d = np.array([[10., 10., 10.]])
        m, (a, b, s, q) = simulate(d, [0.], lambda t, stock: np.array([10.]))
        np.testing.assert_array_equal(s, [[0, 10, 10]])
        np.testing.assert_array_equal(q, [[10, 10, 10]])
        self.assertEqual(m['ready_rate'], 0)
        self.assertEqual(m['fill_rate'], 2 / 3)
        self.assertEqual(m['final_pipeline'], 10)

    def test_fill_rate_not_service_level(self):
        d = np.full((2, 3), 10.)
        m, _ = simulate(d, [10, 10], lambda t, b: np.full(2, 10.))
        self.assertEqual(m['ready_rate'], 0)
        self.assertEqual(m['fill_rate'], 1)

    def test_ceil_and_nonnegative(self):
        _, (_, _, _, q) = simulate(np.ones((2, 2)), [0, 0], lambda t, b: np.array([1.2, -4.]))
        np.testing.assert_array_equal(q, [[2, 2], [0, 0]])

    def test_ratios_and_inventory_average(self):
        d = np.array([[10., 5.], [2., 3.]])
        m, arrays = simulate(d, [15, 8], lambda t, b: np.zeros(2))
        self.assertEqual(m['inventory_demand_ratio'], arrays[1].sum() / d.sum())
        self.assertEqual(m['average_ending_total_inventory'], arrays[1].sum() / 2)
        self.assertEqual(m['average_inventory_per_sku_period'], arrays[1].sum() / 4)

    def test_zero_demand_is_undefined(self):
        m, _ = simulate(np.zeros((2, 2)), [1, 1], lambda t, b: np.zeros(2))
        self.assertIsNone(m['fill_rate'])
        self.assertIsNone(m['inventory_demand_ratio'])

    def test_invalid_order(self):
        with self.assertRaises(ValueError):
            simulate(np.ones((2, 2)), [1, 1], lambda t, b: np.array([np.nan, 0]))

    def test_features_no_future_leakage(self):
        raw = fixture()
        d, x = raw['train_demand'], raw['train_x']
        before = build_supervised_data(d, x, raw['categories'], np.array([30]), False)
        d[:, 30:] += 1e6
        x[:, 30:] -= 1e6
        after = build_supervised_data(d, x, raw['categories'], np.array([30]), False)
        np.testing.assert_array_equal(before.features, after.features)
        self.assertEqual(before.features.shape, (24, 59))

    def test_statistical_forecasts_no_future_leakage(self):
        from replenishment.training.forecast import history_prediction
        raw = fixture()
        d = np.concatenate([raw['train_demand'], raw['test_demand']], 1)
        for name, params in [('baseline', {}), ('naive', {}), ('moving_average', {'window': 6}),
                             ('exponential_smoothing', {'alpha': .5, 'beta': .1}), ('arima', {'p': 1})]:
            a = history_prediction(d.copy(), name, params, 42)[1]
            changed = d.copy()
            changed[:, 60:] *= 1000
            b = history_prediction(changed, name, params, 42)[1]
            np.testing.assert_array_equal(a[:, :49], b[:, :49])

    def test_simplex_and_hull(self):
        from replenishment.training.ensemble import projection
        rng = np.random.default_rng(0)
        w = projection(rng.normal(size=(12, 14)))
        self.assertTrue((w >= 0).all())
        np.testing.assert_allclose(w.sum(1), 1)
        p = rng.uniform(size=(12, 6, 14))
        mixed = np.einsum('ntm,nm->nt', p, w)
        self.assertTrue((mixed >= p.min(2)).all() and (mixed <= p.max(2)).all())

    def test_custom_loss_gradient(self):
        from replenishment.models.xgboost_spo import hybrid_grad_hess
        c = np.array([[1., 2., 4.]])
        p = np.array([[.7, 1.5, 2.]])
        alpha, temperature = .8, .5
        def loss(v):
            logits = (c - 2 * v) / temperature
            maximum = logits.max(1)
            smooth = temperature * (maximum + np.log(np.exp(logits - maximum[:, None]).sum(1)))
            return ((1 - alpha) * .5 * ((v - c) ** 2).mean(1)
                    + alpha * (smooth + 2 * v[:, 0] - c[:, 0])).sum()
        g, h = hybrid_grad_hess(c, p, alpha=alpha, temperature=temperature)
        for j in range(3):
            delta = np.zeros_like(p)
            delta[0, j] = 1e-5
            self.assertAlmostEqual(g[0, j], (loss(p + delta) - loss(p - delta)) / 2e-5, places=5)
        self.assertTrue((h > 0).all())

    def test_all_sources_compile_and_independent(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / 'replenishment').rglob('*.py'):
            text = path.read_text()
            ast.parse(text)
            self.assertNotIn('sys.path', text)
            self.assertNotIn('/Users/', text)
            self.assertNotIn('/opt/homebrew/', text)
            self.assertNotIn('from common.', text)


if __name__ == '__main__':
    unittest.main()
