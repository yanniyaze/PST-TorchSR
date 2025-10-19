from torchsr.datasets import Div2K
from torchsr.models import ninasr_b0, edsr_baseline, rdn_a, carn_m, rcan_g10r20f64
from torchvision.transforms.functional import to_pil_image, to_tensor
import matplotlib.pyplot as plt
import torch
import time

# Dataset laden
dataset = Div2K(root="./scripts/data", scale=2, download=False)
hr, lr = dataset[5]

# Modell laden
inputModel = input("Auswahl: ")
match(str(inputModel).lower()):
    case "ninasr":
        model = ninasr_b0(scale=2, pretrained=True)
    case "edsr":
        model = edsr_baseline(2, True)
    case "rdn":
        model = rdn_a(2,True)
    case "carn":
        model = carn_m(2, True)
    case "rcan":
        model = rcan_g10r20f64(2, True)

model.eval()

start = time.time()
print("Timer start")

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

end = time.time()
print("Timer end ", end - start)
for ax in axs: ax.axis("off")
plt.show() # hr.show(), lr.show(), sr.show()
