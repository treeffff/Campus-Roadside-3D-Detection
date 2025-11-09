import os
from PIL import Image
import numpy as np
import json


def resize_images_and_calib(img_folder, calib_folder, target_w=1920, target_h=1080):
    img_files = [f for f in os.listdir(img_folder) if f.endswith((".png", ".jpg"))]

    for img_name in img_files:
        img_path = os.path.join(img_folder, img_name)
        calib_path = os.path.join(calib_folder, img_name.replace(".jpg", ".json"))

        # 打开图片
        img = Image.open(img_path)
        old_w, old_h = img.size

        # resize 图片
        img_resized = img.resize((target_w, target_h))
        img_resized.save(img_path)

        # 读取 calib
        with open(calib_path, "r") as f:
            calib = json.load(f)

        # 相机内参 cam_K (3x3)
        cam_K = np.array(calib["intrinsic"]).reshape(3, 3)

        # 缩放比例
        scale_x = target_w / old_w
        scale_y = target_h / old_h

        # 修改内参
        cam_K[0, 0] *= scale_x   # fx
        cam_K[1, 1] *= scale_y   # fy
        cam_K[0, 2] *= scale_x   # cx
        cam_K[1, 2] *= scale_y   # cy

        calib["intrinsic"] = cam_K.reshape(-1).tolist()

        # 保存修改后的 json
        with open(calib_path, "w") as f:
            json.dump(calib, f, indent=4)

        print(f"Processed {img_name}, old size: ({old_w},{old_h}) -> new size: ({target_w},{target_h})")

img_folder = "./xfdata/test/images"
calib_folder = "./xfdata/test/calibs"
resize_images_and_calib(img_folder, calib_folder, target_w=1920, target_h=1080)


img_folder = "./xfdata/train/train_images"
calib_folder = "./xfdata/train/train_calibs"
resize_images_and_calib(img_folder, calib_folder, target_w=1920, target_h=1080)

