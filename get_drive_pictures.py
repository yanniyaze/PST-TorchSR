import os
import gdown

folder_id = "1_ZBb3fcsFzG69OGAO-sH6MpMNz8M2pOn"
target_dir = "./scripts/google_drive_pictures"
os.makedirs(target_dir, exist_ok=True)

folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

gdown.download_folder(url=folder_url, output=target_dir, quiet=False, use_cookies=False)

print(f"Alle Bilder wurden in '{target_dir}' heruntergeladen!")
