import os
import json
import shutil
import glob
from tqdm import tqdm

bs80k_root = os.getenv("BS80K_ROOT", "bs80k")
bs80k_new = os.path.join(bs80k_root, "bs80k_wholebody")

os.makedirs(bs80k_new, exist_ok = True)

paths = ["WholeBodyANT","WholeBodyPOST"]

for p in paths:
    for img in tqdm(os.listdir(os.path.join(bs80k_root,"temp", p))):
        if ".jpg" in img:
            shutil.copy(
                os.path.join(bs80k_root, "temp", p, img),
                os.path.join(bs80k_new, img.replace(".jpg",p.replace("WholeBody","")+".jpg")),
                )
