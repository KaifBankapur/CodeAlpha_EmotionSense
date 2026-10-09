"""Inference.

``EmotionPredictor`` loads a checkpoint, rebuilds the exact preprocessing it was
trained with, and returns a probability distribution. The API depends on this
module and never on the training code.
"""
