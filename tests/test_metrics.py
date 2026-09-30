import unittest

from src.metrics import (
    allreduce_bytes_per_step,
    meets_targets,
    optimizer_and_grad_bytes,
    reduction_ratio,
    training_cost_bytes,
)


class MetricsTest(unittest.TestCase):
    def test_optimizer_bytes_scale_with_trainable_params(self):
        self.assertEqual(optimizer_and_grad_bytes(0), 0)
        self.assertEqual(optimizer_and_grad_bytes(100), 1200)

    def test_training_cost_counts_every_step(self):
        self.assertEqual(training_cost_bytes(10, 5), optimizer_and_grad_bytes(10) * 5)

    def test_ring_allreduce_volume(self):
        self.assertEqual(allreduce_bytes_per_step(1000, world_size=1), 0.0)
        # n=2 moves exactly one copy of the fp32 gradient
        self.assertEqual(allreduce_bytes_per_step(1000, world_size=2), 4000.0)
        self.assertAlmostEqual(allreduce_bytes_per_step(1000, world_size=4), 6000.0)

    def test_reduction_ratio(self):
        self.assertAlmostEqual(reduction_ratio(100, 40), 0.60)
        with self.assertRaises(ValueError):
            reduction_ratio(0, 0)

    def test_target_gate_boundaries(self):
        passed = meets_targets(100, 40, 0.90, 0.89, target_reduction=0.60, max_accuracy_drop=0.01)
        self.assertTrue(passed["passed"])
        self.assertTrue(passed["cost_ok"])
        self.assertTrue(passed["quality_ok"])

        short_on_cost = meets_targets(100, 41, 0.90, 0.90)
        self.assertFalse(short_on_cost["cost_ok"])
        self.assertFalse(short_on_cost["passed"])

        quality_miss = meets_targets(100, 10, 0.90, 0.889, max_accuracy_drop=0.01)
        self.assertFalse(quality_miss["quality_ok"])
        self.assertFalse(quality_miss["passed"])

        better_accuracy = meets_targets(100, 5, 0.80, 0.84)
        self.assertTrue(better_accuracy["passed"])
        self.assertLess(better_accuracy["accuracy_drop"], 0)


if __name__ == "__main__":
    unittest.main()
