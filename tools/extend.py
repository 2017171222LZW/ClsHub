from pathlib import Path
from typing import Any

import numpy
from PIL import Image
from PIL import ImageFilter
import numpy as np
import uuid


def traverse(path: Any):
    root = Path(path)
    for file in root.glob("*/*"):
        yield file


def createPath_R(path: str):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)


def saveAs(path: Any, new_path: Any):
    if type(path) != Path or type(new_path) != Path:
        path = Path(path)
        new_path = Path(new_path)
    if new_path.exists():
        print(f"{new_path} file is exist.")
        return None
    else:
        createPath_R(new_path.parent)
    with open(path, "rb") as f1, open(new_path, 'wb') as f2:
        data = f1.read()
        f2.write(data)


def openImg(path: Any):
    path = Path(path)
    try:
        img = Image.open(path, 'r')
        img = np.array(img)
        return img
    except Exception as e:
        print(e, f"\n\t file: {path}")


def randID():
    id = uuid.uuid4()
    id = str(id).split('-')[-1]
    return '_' + id


def cropper(img: numpy.ndarray, top_discard=0.1):
    w, h, c = img.shape
    img = img[:, int(top_discard * h):, :]
    w, h, c = img.shape
    croped_imges_wh = [
        # [0, w // 2, 0, h // 2], [0, w // 2, h // 2, h], [w // 2, w, 0, h // 2], [w // 2, w, h // 2, h]
        [w // 6 * 1, w // 6 * 5, h // 6 * 1, h // 6 * 5],
    ]
    croped_imges = []
    for args in croped_imges_wh:
        croped_imges.append(img[args[0]:args[1], args[2]:args[3], :])

    croped_imges = [Image.fromarray(e) for e in croped_imges] + \
                   [Image.fromarray(e).rotate(90) for e in croped_imges] + \
                   [Image.fromarray(e).rotate(45) for e in croped_imges] + \
                   [Image.fromarray(e).rotate(135) for e in croped_imges]
                   # [Image.fromarray(e).filter(ImageFilter.BLUR) for e in croped_imges]
# [Image.fromarray(e).filter(ImageFilter.GaussianBlur(radius=2)) for e in croped_imges] + \
# [Image.fromarray(e).filter(ImageFilter.MedianFilter(size=3)) for e in croped_imges] + \


    return croped_imges


def saveImg(data: Image.Image, path):
    path = Path(path)
    # print(path)
    return
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    print(path)
    if not path.exists():
        data.save(path)
    else:
        print(f"file {path} is exist.")


def rename(file: Any, dir: str):
    file = Path(file)
    name = file.name.split('.')
    name[-2] += randID()
    name = '.'.join(name)
    name = str(file.parent / name).replace(dir, '')
    name = Path(name)
    return name


def cropDir(dir: Any, dst: Any):
    dir = Path(dir)
    for file in traverse(dir):
        print(file)
        data = openImg(file)
        if data is None or len(data.shape) < 3:
            print(f'file: {file} open error!')
            continue
        imges = cropper(data)
        for img in imges:
            new_file = dst + str(rename(file, str(dir)))
            new_file = Path(new_file)
            if not new_file.parent.exists():
                continue
                new_file.parent.mkdir(parents=True, exist_ok=True)
            # saveImg(img, new_file)


if __name__ == '__main__':
    # path = r'E:\数据\6C_train'
    # # saveAs("./dir1/26.jpg", "./dir2/26.jpg")

    # data = openImg("./dir1/126.png")
    # data = np.array(data)
    # print(len(data.shape))
    # data = data[:, :, None]
    # print(data.shape)

    # imges = cropper(data)
    # print(type(imges[0]))
    # rename("./dir1/26.jpg")
    cropDir(r'/DATA/lzw/data/10C/', r'/DATA/lzw/data/10C-extend/')
    # for f in traverse(r'E:\数据\6C_train'):
    #     print(f)
