"""Tests for optional ONNX Runtime CPU budgets."""

import unittest

import onnxruntime as ort

from frigate.domain.detectors.plugins.onnx import ONNXDetectorConfig, create_session_options


class ONNXThreadOptionsTest(unittest.TestCase):
    def test_defaults_preserve_upstream_session_behavior(self) -> None:
        config = ONNXDetectorConfig.model_construct(
            intra_op_num_threads=None,
            inter_op_num_threads=None,
            allow_spinning=None,
            execution_mode=None,
        )
        self.assertIsNone(create_session_options(config))

    def test_explicit_cpu_budget_sets_all_session_options(self) -> None:
        config = ONNXDetectorConfig.model_construct(
            intra_op_num_threads=2,
            inter_op_num_threads=1,
            allow_spinning=False,
            execution_mode="sequential",
        )
        options = create_session_options(config)
        self.assertIsNotNone(options)
        self.assertEqual(options.intra_op_num_threads, 2)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(
            options.get_session_config_entry("session.intra_op.allow_spinning"),
            "0",
        )
        self.assertEqual(
            options.get_session_config_entry("session.inter_op.allow_spinning"),
            "0",
        )


if __name__ == "__main__":
    unittest.main()
