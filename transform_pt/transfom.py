
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from models.nets.model_mobilenet_v2 import get_pose_net


from models.config.config import config
from utils import torch_tools

device = "cuda:0"
config.MODEL.EXTRA.NUM_DECONV_FILTERS = [128, 128, 128]
config.MODEL.EXTRA.WIDTH_MULT = 1
model = get_pose_net(config, width_mult=config.MODEL.EXTRA.WIDTH_MULT,is_train=False)
# model = get_pose_net(config,is_train=False)
tempmodel = torch.load("E:\python_project\HKD_mobilenetv2\cfg\model_mobilenet_v2_1.0_17_192_256_128_gaussian_45_1.25_coco_2024-03-13-22-48\model\model_000_0.0003.pth", map_location={'cuda:5': 'cuda:0'})
# tempmodel.pop('final_layer.weight')
# tempmodel.pop('final_layer.bias')
model.load_state_dict(tempmodel, strict=False)                #改需要转的文件
# model = torchvision.models.mobilenet_v3_small(pretrained=True)
model.eval()
example = torch.rand(1, 3, 192, 256)#(batchsize,channels,高，宽)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model_000_0.0005.pt")    #转成pt格式，lzl任意起的名字
print(torch.__version__)

