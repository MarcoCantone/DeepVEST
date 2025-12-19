from monai.metrics import DiceMetric

class DiceMIP(DiceMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_pred, y, **kwargs):
        return super().__call__([x.max(-1).values for x in y_pred], y, **kwargs)

