# Drosophila 2D Pose

- Load the model.
```python
from model import Drosophila2DPose
from parser import create_parser

checkpoint_path = '/home/user/Desktop/DeepFly3D/weights/sh8_deepfly.tar'
args = create_parser().parse_args('')
model = Drosophila2DPose(checkpoint_path=checkpoint_path, **args.__dict__).cuda()
```

- Load the data.
```python
from inference import path2inp
from dataset import Drosophila2Dataset
from torch.utils.data import DataLoader

image_path = '/home/user/Desktop/DeepFly3D/data/test/'
inp = path2inp(image_path) # extract list of images under the folder
dat = DataLoader(Drosophila2Dataset(inp), batch_size=8)
```

- Do the inference.
```python
from inference import inference
points2d = inference(model, dat)
```