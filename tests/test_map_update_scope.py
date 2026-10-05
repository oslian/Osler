import unittest
import torch
from scripts.audit_map_update_scope import MapUpdateFilter


class MapUpdateScopeTest(unittest.TestCase):
    def test_nonmap_events_and_rate_events_do_not_update(self):
        torch.manual_seed(1)
        model = MapUpdateFilter(9, 8, 3).eval()
        x = torch.randn(1, 2, 9)
        x[..., 3] = 1; x[..., -1] = torch.tensor([0., .5])
        extra = torch.randn(1, 1, 9); extra[..., 3] = 0
        extra[..., -2] = 1; extra[..., -1] = .25
        full = torch.cat([x[:, :1], extra, x[:, 1:]], dim=1)
        a = model(x, torch.ones(1, 2, dtype=torch.bool))
        b = model(full, torch.ones(1, 3, dtype=torch.bool))
        for left, right in zip(a, b): torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == '__main__': unittest.main()
