"""
Emotion Recognition from Speech - machine learning package.

Pipeline
--------
RAVDESS audio  ->  audio preprocessing  ->  MFCC features  ->  CNN+LSTM  ->  softmax

The single source of truth for emotion labels, preprocessing parameters and
feature-extraction parameters lives in :mod:`ml.config`. The very same objects
are used at training time and at inference time, which is what guarantees that
the API can never silently drift away from the model it serves.
"""

__version__ = "1.0.0"
