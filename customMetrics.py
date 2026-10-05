"""Custom metrics."""

from monai.metrics import DiceMetric


class DiceMIP(DiceMetric):
    """Dice similarity coefficient computed in maximum intensity projection (MIP) space.

    The (one-hot) 3D prediction is projected along the last (axial, S) axis with a maximum
    operation and compared with a 2D reference annotation drawn on the post-contrast MIP.
    It is used during training of the vessel-removal model to select the checkpoint with the
    best MIP-space performance (``best_model_mip.pth``).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred, y, **kwargs):
        return super().__call__([x.max(-1).values for x in y_pred], y, **kwargs)
