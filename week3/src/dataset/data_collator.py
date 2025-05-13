import torch 
import torch.nn as nn

class CustomDataCollator:
    def __init__(self, args):
        self.args = args

    def __call__(self, features):
        # Unzip the batch
        image = [torch.tensor(feature["image"]) for feature in features]
        image = torch.stack(image).to(torch.float32)

        label = [torch.tensor(feature["label"]) for feature in features]
        label = torch.stack(label).to(torch.long)

        return {
            'image' : image,
            'label' : label
        }