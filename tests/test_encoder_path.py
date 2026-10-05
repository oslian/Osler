import unittest
from pathlib import Path
import torch
from scripts.audit_encoder_path import probe
from whole_body_jepa.runtime import load_release


class EncoderPathTest(unittest.TestCase):
    def test_release_context_is_connected_but_no_separate_organ_transition(self):
        torch.set_num_threads(1)
        assets = load_release(Path('models'))
        result = probe(assets.model, assets.checkpoint['variables'].index('glucose'))
        self.assertFalse(result['teacher_parameters_trainable'])
        self.assertEqual(result['separate_organ_argument_max_effect'], 0)
        for item in result['gradients'].values():
            self.assertGreater(item['non_target_current_norm'], 0)
            self.assertGreater(item['non_target_history_norm'], 0)
            self.assertTrue(item['variable_mixer_active'])
            self.assertTrue(item['module_mixer_active'])
            self.assertTrue(item['belief_filter_active'])
