from pathlib import Path
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
from torchsr.models import ninasr_b0
import matplotlib.pyplot as plt

# 1️⃣ Pfad zu deinem Google Drive Ordner (lokal heruntergeladen)
image_dir = Path("./scripts/google_drive_pictures/Yannik")
image_paths = list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))

print(f"Gefundene Bilder: {len(image_paths)}")

# 2️⃣ Modell laden
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ninasr_b0(scale=2, pretrained=True).to(device)
model.eval()

# 3️⃣ Schleife über Bilder
for img_path in image_paths:
    lr = Image.open(img_path).convert("RGB")

    # SR ausführen
    lr_t = to_tensor(lr).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = model(lr_t)
    sr = to_pil_image(sr_t.squeeze(0).cpu().clamp(0, 1))

    # Ergebnis anzeigen
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(lr)
    axs[0].set_title("Original / LR")
    axs[0].axis("off")

    axs[1].imshow(sr)
    axs[1].set_title("Super-Resolved")
    axs[1].axis("off")

    plt.suptitle(img_path.name)
    plt.show()
