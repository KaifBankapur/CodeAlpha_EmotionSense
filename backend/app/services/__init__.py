"""Service layer.

Owns the parts with real logic: reading an upload without touching the disk,
converting validation failures into typed errors, and serialising access to the
shared model instance.
"""
