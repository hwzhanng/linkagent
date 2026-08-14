"""Minimal no-op debug-draw compatibility module.

The tomato task does not call Isaac Sim's optional debug-draw API.  The local
Isaac Sim pip installation omits that package, but Kit still declares it as an
extension dependency.  Keeping the importable namespace lets the simulator
load the actual PhysX/rendering extensions without changing task behavior.
"""
