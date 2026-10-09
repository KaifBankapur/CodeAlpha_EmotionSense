"""Feature extraction.

One extractor, used identically at training and inference time. If this package
grew a second implementation, the API would quietly stop matching the model it
was trained with - which is the single most damaging failure mode here.
"""
