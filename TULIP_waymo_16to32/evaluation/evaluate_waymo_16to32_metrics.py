"""Range/intensity/mask metrics scaffold."""


def evaluate_sample(prediction, target, mask=None):
    """Compute metrics without assuming final normalization yet."""
    raise NotImplementedError
