from torchsr.datasets import Div2K
from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt
import torch

# Dataset laden
dataset = Div2K(root="./scripts/data", scale=2, download=False)
hr, lr = dataset[5]

# Modell laden
model = ninasr_b0(scale=2, pretrained=True)
model.eval()

# Super-Resolution ausführen
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
lr_t = to_tensor(lr).unsqueeze(0).to(device)
with torch.no_grad():
    sr_t = model(lr_t)
sr = to_pil_image(sr_t.squeeze(0).cpu().clamp(0, 1))

# Bilder anzeigen
fig, axs = plt.subplots(1, 3, figsize=(12, 6))
axs[0].imshow(lr)
axs[0].set_title("Low-Resolution")
axs[1].imshow(sr)
axs[1].set_title("Super-Resolved")
axs[2].imshow(hr)
axs[2].set_title("High-Resolution")
for ax in axs: ax.axis("off")
sr.show() # hr.show(), lr.show(), sr.show()
