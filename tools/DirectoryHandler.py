from pathlib import Path
import os
import shutil

from PIL import Image

path = Path('/DATA/lzw/data/10C/')
limit_num = 500
topk = 50

# print("统计样本数 ≥ ", limit_num, " 的岩石类别")
lists = []
total = 0
count = 0

def IsValidImage(pathfile):
  bValid = True
  try:
    Image.open(pathfile).verify()
  except:
    bValid = False
  return bValid

for d in path.iterdir():
    if d.is_dir() and len(list(d.iterdir())) >= limit_num:
        # print(d.name, ': ' , len(list(d.iterdir())))
        count += len(list(d.iterdir()))
        for f in d.iterdir():
            if not IsValidImage(f):
                print("文件不完整: ", f)
                os.remove(f)
    else:
        # shutil.rmtree(d)
        total += len(list(d.iterdir()))
    lists.append({"name": d.name, "length": len(list(d.iterdir()))})

print(f"------------top{topk}----------------")
lists.sort(key=lambda info: info["length"], reverse=True)
for p in lists[:topk]:
    print(p)
print("total useful images: ", count)
total += count
print("total images: ", total)


