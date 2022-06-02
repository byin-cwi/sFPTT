import numpy as np
from PIL import Image
import torch

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   获得类
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   获得先验框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

#---------------------------------------------------#
#   load weights
#---------------------------------------------------#
def load_weight_layer(layer_i,name_i, state_dict):
    with torch.no_grad():
        layer_i.conv1_x.weight.copy_(state_dict[name_i+'.conv.weight'])
        layer_i.BN.weight.copy_(state_dict[name_i+'.bn.weight'])
        layer_i.BN.bias.copy_(state_dict[name_i+'.bn.bias'])
        layer_i.BN.running_mean.copy_(state_dict[name_i+'.bn.running_mean'])
        layer_i.BN.running_var.copy_(state_dict[name_i+'.bn.running_var'])
    
def load_weights(file_path,model):
    state_dict = torch.load(file_path)
    with torch.no_grad():
        model.network.conv_P5_yolo_1.weight.copy_(state_dict['yolo_headP5.1.weight'])
        model.network.conv_P5_yolo_1.bias.copy_(state_dict['yolo_headP5.1.bias'])

        model.network.conv_P5_yolo_2.weight.copy_(state_dict['yolo_headP4.1.weight'])
        model.network.conv_P5_yolo_2.bias.copy_(state_dict['yolo_headP4.1.bias'])
        
    load_weight_layer(model.network.conv_P5,'conv_for_P5',state_dict)
    load_weight_layer(model.network.conv_yolo_1,'yolo_headP5.0',state_dict)
    load_weight_layer(model.network.conv_yolo_2, 'yolo_headP4.0',state_dict)
    load_weight_layer(model.network.conv_up,'upsample.upsample.0',state_dict)

    load_weight_layer(model.network.conv1,'backbone.conv1',state_dict)
    load_weight_layer(model.network.conv2,'backbone.conv2',state_dict)
    load_weight_layer(model.network.conv3,'backbone.conv3',state_dict)


    load_weight_layer(model.network.block1.conv1,'backbone.resblock_body1.conv1',state_dict)
    load_weight_layer(model.network.block1.conv2,'backbone.resblock_body1.conv2',state_dict)
    load_weight_layer(model.network.block1.conv3,'backbone.resblock_body1.conv3',state_dict)
    load_weight_layer(model.network.block1.conv4,'backbone.resblock_body1.conv4',state_dict)

    load_weight_layer(model.network.block2.conv1,'backbone.resblock_body2.conv1',state_dict)
    load_weight_layer(model.network.block2.conv2,'backbone.resblock_body2.conv2',state_dict)
    load_weight_layer(model.network.block2.conv3,'backbone.resblock_body2.conv3',state_dict)
    load_weight_layer(model.network.block2.conv4,'backbone.resblock_body2.conv4',state_dict)

    load_weight_layer(model.network.block3.conv1,'backbone.resblock_body3.conv1',state_dict)
    load_weight_layer(model.network.block3.conv2,'backbone.resblock_body3.conv2',state_dict)
    load_weight_layer(model.network.block3.conv3,'backbone.resblock_body3.conv3',state_dict)
    load_weight_layer(model.network.block3.conv4,'backbone.resblock_body3.conv4',state_dict)

    return model