import unittest
import numpy as np

from whole_body_jepa.ragged_native_batch import RaggedNativeBatch
from whole_body_jepa.native_event_encoder import native_arrays
from scripts.train_availability_encoder import batch
from whole_body_jepa.ragged_native_transform import transformed_storage
from whole_body_jepa.availability_inputs import apply_numeric_rescale, restrict_input_support
from whole_body_jepa.history_event_input import prepare_events, append_events


def fixture():
    x = np.arange(8 * 9, dtype=np.float32).reshape(8, 9)
    base = dict(input=x, current=x[:, :2], mask=x[:, 2:4], ages=x[:, 4:6],
        presence=x[:, 7:8], horizon=np.full(8, 6., np.float32),
        query_rows=np.array([4, 5]), global_query_rows=np.array([1, 0]))
    native = dict(input=x[:3], event_values=np.array([[0., np.nan, 0.], [1., 0., 2.], [0., 1., 2.]]),
        history_indices=np.array([0, 2, 1]), query_ptr=np.array([0, 2, 3]),
        global_query_rows=np.array([1, 0]))
    return base, native


class RaggedBatchTest(unittest.TestCase):
    def test_transform_matches_dense_including_padding(self):
        base, native = fixture()
        compact = RaggedNativeBatch([(base, native)], 2)
        dense, rows, values = native_arrays(base, native, 2, 1)
        center = np.array([0.5, 0.], np.float32)
        scale = np.array([2., 1.], np.float32)
        modules = np.array([0, 0])
        original = compact.input.copy()
        for own in (False, True):
            allowed = np.array([True, not own])
            transformed, ec, es = transformed_storage(compact, center, scale, allowed,
                modules, np.array([1]), own=own)
            features, old_ec, old_es = prepare_events(values, np.unique(dense['history'][rows[:1]]))
            if own:
                features[:] = 0
            expected = append_events(restrict_input_support(
                apply_numeric_rescale(dense, center, scale), allowed, modules), features)
            for a, b in zip(transformed.batch([1, 0]), batch(expected, rows)):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
            np.testing.assert_array_equal(ec, old_ec)
            np.testing.assert_array_equal(es, old_es)
        np.testing.assert_array_equal(compact.input, original)

    def test_exact_dense_equivalence_and_global_order(self):
        base, native = fixture()
        compact = RaggedNativeBatch([(base, native)], 2)
        dense, rows, _ = native_arrays(base, native, 2, 1)
        for a, b in zip(compact.batch([1, 0]), batch(dense, rows)):
            np.testing.assert_array_equal(a.numpy(), b.numpy())
        np.testing.assert_array_equal(compact.event_values[1:], native['event_values'])

    def test_incomplete_duplicate_and_bad_pointer_rejected(self):
        base, native = fixture()
        with self.assertRaises(ValueError):
            RaggedNativeBatch([(base, native)], 3)
        with self.assertRaises(ValueError):
            RaggedNativeBatch([(base, native), (base, native)], 2)
        native['query_ptr'] = np.array([0, 2, 4])
        with self.assertRaises(ValueError):
            RaggedNativeBatch([(base, native)], 2)


if __name__ == '__main__':
    unittest.main()
