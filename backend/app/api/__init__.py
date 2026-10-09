"""HTTP layer: routers and typed errors.

Routes translate HTTP into service calls and back again. They hold no ML logic,
so the request contract can be reasoned about without loading a model.
"""
