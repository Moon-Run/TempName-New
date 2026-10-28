import unittest
from routing import module_routes


class RoutingTests(unittest.TestCase):
    def observations(self):
        rows = {}
        for i in range(12):
            for suffix, k in [('self_attention.linear_proj', 2048), ('mlp.linear_fc2', 8192)]:
                rows[f'layers.{i}.{suffix}'] = dict(M=2048, N=2048, K_global=k, K_local=k//4)
        return rows

    def test_candidate_changes_only_mlp(self):
        routes = module_routes(self.observations(), 'mlp_remote')
        self.assertEqual(list(routes.values()).count('remote_first'), 12)
        self.assertTrue(all(v == 'original' for k, v in routes.items() if '.self_attention.' in k))

    def test_baselines(self):
        for p in ['native', 'original']:
            self.assertEqual(set(module_routes(self.observations(), p).values()), {p})

    def test_arrival_changes_only_mlp(self):
        routes = module_routes(self.observations(), 'mlp_remote_arrival')
        self.assertEqual(list(routes.values()).count('remote_arrival'), 12)
        self.assertTrue(all(v == 'original' for k, v in routes.items() if '.self_attention.' in k))

    def test_wrong_shape_cannot_silently_select_remote_attention(self):
        rows = self.observations()
        rows['layers.0.self_attention.linear_proj'].update(K_global=8192, K_local=2048)
        for policy in ('mlp_remote', 'mlp_remote_arrival'):
            with self.assertRaises(AssertionError):
                module_routes(rows, policy)

    def test_missing_layer_rejected(self):
        rows = self.observations()
        del rows['layers.5.mlp.linear_fc2']
        with self.assertRaises(AssertionError):
            module_routes(rows, 'mlp_remote')

    def test_unexpected_module_rejected(self):
        rows = self.observations()
        rows['unrelated.linear_proj'] = rows['layers.0.self_attention.linear_proj']
        with self.assertRaises(AssertionError):
            module_routes(rows, 'mlp_remote')


if __name__ == '__main__':
    unittest.main()
