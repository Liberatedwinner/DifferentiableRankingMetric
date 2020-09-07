import torch

x = torch.load("best_res/epinion-712/torch-warp", map_location='cpu')

for a, b in x.items():
    if "_at_" in a:
        for k, v in b[0].items():
            print(v)
