import os
import torch
import torch.nn as nn
try:
    from . import block as B
except ImportError:
    import block as B

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=4, num_channels=3, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        # 1. Feature Extraction
        self.feature_extraction = B.conv_block(num_channels, d, kernel_size=5, padding=2)

        # 2. Shrinking
        self.shrinking = B.conv_block(d, s, kernel_size=1)

        # 3. Non-linear Mapping 
        mapping_layers = []
        for _ in range(m):
            mapping_layers.extend([
                nn.Conv2d(s, s, kernel_size=3, padding=1),
                nn.PReLU(s)
            ])
        self.mapping = nn.Sequential(*mapping_layers)

        # 4. Expanding
        self.expanding = B.conv_block(s, d, kernel_size=1)

        # 5. Deconvolution (Upscaling)
        self.deconv = B.deconv_block(d, num_channels, scale_factor=scale_factor)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrinking(x)
        x = self.mapping(x)
        x = self.expanding(x)
        x = self.deconv(x)
        return x

def load_model(weights_path, scale_factor=4, device='cpu'):
    model = FSRCNN(scale_factor=scale_factor, num_channels=3).to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model