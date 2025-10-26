import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
from torchsr.models import ninasr_b0
import matplotlib.pyplot as plt

fileIds = [
    "1Y9YDk-SW7wRx2vzKGi71LYN5LMKSMhYc",
    "1MBvuRVn-iCprUtEgp7dVE0WWiEZ5F5rk",
    "1Wxxs3pOkUANkAGI57K_T6w32XgRy0Pn9"
]

file_id = fileIds[0]
url = f"https://drive.google.com/uc?id={file_id}"

# Bild direkt aus Drive laden
response = requests.get(url)
response.raise_for_status()

lr_img = Image.open(BytesIO(response.content)).convert("RGB")

# === Modell laden ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ninasr_b0(scale=8, pretrained=True).to(device)
model.eval()

# === Super-Resolution ===
lr_t = to_tensor(lr_img).unsqueeze(0).to(device)
with torch.no_grad():
    sr_t = model(lr_t)
sr_img = to_pil_image(sr_t.squeeze(0).cpu().clamp(0, 1))

# === Anzeige ===
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(lr_img)
axs[0].set_title("Original / LR")
axs[0].axis("off")

axs[1].imshow(sr_img)
axs[1].set_title("Super-Resolved")
axs[1].axis("off")

lr_img.show()
sr_img.show()