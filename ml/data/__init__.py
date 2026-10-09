"""Dataset layer: RAVDESS parsing, speaker-disjoint splits, features, caching.

``ml.data.ravdess`` owns the file-naming grammar and the split policy;
``ml.data.preprocessing`` owns everything that happens to a waveform;
``ml.data.features`` and ``ml.data.dataset`` own MFCC extraction and the cache.
``ml.data.loading`` is the single entry point every script uses, so no two
experiments can accidentally disagree about the data.
"""
