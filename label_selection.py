import numpy as np
from torch.autograd import Variable
import torch as torch

def choose_target(image, net, Matrix_path, target_path, delta_1=0.8, delta_2=1e-2, classes_num=1000):

    """
       :param image: Image of size HxWx3
       :param net: network to be attacked (input: images, output: values of activation **BEFORE** softmax)
       :param Matrix_path: path for Distance Matrix_D,
       :param target_path: path for target vector, target[i] records the nearest label to i according to Matrix_D
       :param delta_1, delta_2: hyperparameters
       :param classes_num: the number of classes in the dataset. For example, ImageNet has 1000 classes.
       :return: target labels under LabelFool
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    # Load Matrix
    target = np.load(target_path)
    Matrix_D = np.load(Matrix_path)

    with torch.no_grad():
        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True))
        f_image_soft = torch.nn.functional.softmax(f_image)
        f_image_new, I = torch.sort(f_image_soft, descending=True)
        f_image_new = f_image_new.squeeze()
        I = I.squeeze()

    # If \hat p_1 > \delta_1, choose the nearest label in Matrix D
    if f_image_new[0] > delta_1:
        target_label = int(target[I[0]])

    # If \hat p_1 <= \delta_1
    else:
        # Find S={j:\hat p_j > \delta_2}
        for i in range(classes_num):
            if f_image_new[i] < delta_2:
                flag = i
                break
        # A special case where the max confidence is smaller than \delta_2
        if flag == 0:
            flag = 1

        # label vector: the set S
        label_vector = I[0:flag]
        # confidence vector: confidence for elements in S
        confidence_vector = f_image_new[0:flag]

        # The weighted distance
        # Distance_vector[i] = \sum_{j\in S} \hat p_j * D_i,j
        Distance_vector = torch.zeros(classes_num)
        for i in range(classes_num):
            Distance = torch.zeros(flag)
            for j in range(flag):
                Distance[j] = Matrix_D[i, int(label_vector[j])]
            Distance_vector[i] = torch.sum(confidence_vector.cpu() * Distance.cpu())

        target_label = np.argmin(Distance_vector.cpu().numpy())

    return target_label

