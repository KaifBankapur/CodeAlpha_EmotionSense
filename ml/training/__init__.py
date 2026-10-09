"""Training and evaluation.

``trainer`` owns the loop, checkpointing and best-model selection; ``evaluate``
owns the held-out evaluation that is run exactly once, after selection is
frozen.
"""
