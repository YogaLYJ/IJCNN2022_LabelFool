import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

def attack_to_target(image, net, target_label, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network to be attacked (input: images, output: values of activation **BEFORE** softmax)
       :param target_label
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, perturbed image, flag
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")

    with torch.no_grad():
        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True))
        f_image_soft = torch.nn.functional.softmax(f_image)
        f_image_new, I = torch.sort(f_image_soft, descending=True)
        I = I.squeeze()

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = int(I[0])

    while k_i != target_label and loop_i < max_iter:

        fs[0, k_i].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        zero_gradients(x)

        fs[0, target_label].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        f_k = (fs[0, target_label] - fs[0, k_i]).data.cpu().numpy()

        pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

        pert = pert_k
        w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    if k_i == target_label:
        flag = True
    else:
        flag = False

    return r_tot, loop_i, pert_image, flag
