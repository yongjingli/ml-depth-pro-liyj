import numpy as np
from tqdm import tqdm
from PIL import Image
import depth_pro
import os
import torch


def infer_depth_pro():
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device=torch.device("cpu"))
    model.eval()

    root = "/home/pxn-lyj/Egolee/data/test/pose_shi"
    img_root = os.path.join(root, "colors")
    depth_root = os.path.join(root, "depths")

    img_names = [name for name in os.listdir(img_root) if name[-4:] in [".jpg", ".png"]]
    img_names = list(sorted(img_names, key=lambda x: int(x.split(".")[0].split("_")[0])))

    for img_name in tqdm(img_names):
        print(img_name)
        img_path = os.path.join(img_root, img_name)

        # Load and preprocess an image.
        image, _, f_px = depth_pro.load_rgb(img_path)
        image = transform(image)

        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        focallength_px = prediction["focallength_px"]

        ing_mask_path = os.path.join(depth_root, img_name.replace("_color.jpg", "_depth.npy"))
        np.save(ing_mask_path, depth)



if __name__ == "__main__":
    print("Start")
    infer_depth_pro()
    print("End")
