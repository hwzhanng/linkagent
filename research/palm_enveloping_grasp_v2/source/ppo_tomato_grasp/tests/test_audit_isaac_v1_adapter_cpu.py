from __future__ import annotations

import unittest

from simulation.ppo_tomato_grasp.audit_isaac_v1_adapter_cpu import audit


class IsaacV1AdapterAuditTests(unittest.TestCase):
    def test_small_fuzz_audit_passes_without_cuda_or_hardware(self):
        report = audit(64, 9)
        self.assertTrue(report["passed_cpu_scope"])
        self.assertFalse(report["cuda_used"])
        self.assertFalse(report["hardware_command_allowed"])

    def test_invalid_sample_count_fails_closed(self):
        with self.assertRaises(ValueError):
            audit(31, 9)


if __name__ == "__main__":
    unittest.main()
