import os
import torch
import torchvision.transforms as T
import torchvision
'''
Load Models
'''
def load_model(name,pretrained=True):
    if name=='resnet50':
        model=torchvision.models.resnet50(pretrained=pretrained)
    elif name=='resnet152':
        model=torchvision.models.resnet152(pretrained=pretrained)
    elif name=='vgg16':
        model=torchvision.models.vgg16(pretrained=pretrained)
    elif name=='alexnet':
        model=torchvision.models.alexnet(pretrained)
    elif name=='squeezenet':
        model=torchvision.models.squeezenet1_1(pretrained)
    elif name=='densenet':
        model=torchvision.models.densenet121(pretrained)
    else:
        raise NotImplementedError('{} is not allowed!'.format(name))

    return model

'''
Save Images
'''
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_res(save_dir, filename,pert_img):

    ensure_dir(save_dir)

    saved_file='{}/{}.png'.format(save_dir,filename[:-5])

    # save by Image.save
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def clip_tensor(A, minv, maxv):
        A = torch.max(A, minv * torch.ones(A.shape))
        A = torch.min(A, maxv * torch.ones(A.shape))
        return A

    clip = lambda x: clip_tensor(x, 0, 255)

    tf = T.Compose([T.Normalize(mean=[0, 0, 0], std=list(map(lambda x: 1 / x, std))),
                    T.Normalize(mean=list(map(lambda x: -x, mean)), std=[1, 1, 1]),
                    T.Lambda(clip),
                    T.ToPILImage()])

    pert_img = tf(pert_img.cpu()[0])
    pert_img.save(saved_file)
