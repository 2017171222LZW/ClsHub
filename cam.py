import pathlib
from torch.nn import Linear
from models.get_model import get
# Define your model
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp,GradCAM,GradCAMpp,XGradCAM,LayerCAM

enum = ['大理岩.jpg', '橄榄.jpg', '泥岩.jpg', '玄武岩.jpg', '砾岩.jpg', '花岗岩.jpg']

path = "images/swin_s_liyan/xgc-liyan.jpg"
ckpt = "output/rocks_extend/swin_s/checkpoint-best.pth"
img_path = 'images/'
img_name = enum[4]

if __name__ == '__main__':
    model = get(ckpt).eval()
    for name, param in model.named_parameters():
        print(name)
    cam_extractor = XGradCAM(model, target_layer="model.features.6")
    img = read_image(img_path + img_name)

    # Preprocess it for your chosen model 0.485, 0.456, 0.406
    input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)

    # Preprocess your data and feed it to the model
    out = model(input_tensor)
    # Retrieve the CAM by passing the class index and the model output
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    import matplotlib.pyplot as plt
    from torchcam.utils import overlay_mask

    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    if not pathlib.Path(path).parent.exists():
        pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    result.save(path)
