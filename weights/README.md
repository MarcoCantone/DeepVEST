# Trained models

| File | Model | Original checkpoint |
|---|---|---|
| `deepvest_segmentation.pth` | 3D vessel segmentation (pre-contrast + 2nd post-contrast) | `seed34_AttentionUnet/best_model.pth` |
| `deepvest_removal.pth` | vessel removal (2nd post-contrast, MIP-space selection) | `seed34_post2_selectionMIP/best_model_mip.pth` |

Both are PyTorch `state_dict`s of `monai.networks.nets.AttentionUnet` (see `configs/`) and are loaded with:

```python
from config import load_config
from utilities import load_model
model = load_model(load_config("configs/deepvest_segmentation.yaml"), "weights/deepvest_segmentation.pth", "cuda")
```
