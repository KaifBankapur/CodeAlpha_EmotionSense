"""Model definitions.

Currently one architecture: a 2-D CNN over MFCC planes, a BiLSTM over time, then
masked additive-attention pooling. Kept small deliberately - on this dataset a
larger model fits speaker identity rather than emotion.
"""
