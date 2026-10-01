"""Audited v3 palm-supported five-finger enclosure experiment.

The v3 package never sends commands to hardware.  It adds immutable scenario
manifests, run provenance, strict per-condition telemetry, and event-driven
control on top of the archived v2 simulation implementation.
"""

from .scenario import ScenarioError, canonical_hash, load_scenario

__all__ = ["ScenarioError", "canonical_hash", "load_scenario"]
