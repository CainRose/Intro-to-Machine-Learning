import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import torch.nn as nn
import torchvision
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from torch.autograd import Variable
from torch import autograd
from copy import deepcopy
import cv2

from faces import train_classifier, get_error, weight_init
from deepfaces import MyAlexNet, load_data
import deepfaces

import sys
from pytorch_cnn_visualizations.src.guided_backprop import GuidedBackprop
from pytorch_cnn_visualizations.src.vanilla_backprop import VanillaBackprop
from pytorch_cnn_visualizations.src.gradcam import GradCam
from pytorch_cnn_visualizations.src.guided_gradcam import guided_grad_cam
from pytorch_cnn_visualizations.src.inverted_representation import \
    InvertedRepresentation
from pytorch_cnn_visualizations.src.generate_class_specific_samples import \
    ClassSpecificImageGeneration
from pytorch_cnn_visualizations.src.misc_functions import save_gradient_images, \
    convert_to_grayscale, get_positive_negative_saliency, \
    save_class_activation_on_image

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


class FaceModel(torch.nn.Module):
    def __init__(self, num_classes=1000):
        super(FaceModel, self).__init__()

        alexnet = MyAlexNet()
        alexnet.eval()
        self.features = torch.nn.Sequential(
            alexnet.features[0], alexnet.features[1], alexnet.features[2],
            # Conv1
            alexnet.features[3], alexnet.features[4], alexnet.features[5],
            # Conv2
            alexnet.features[6], alexnet.features[7],  # Conv3
            alexnet.features[8], alexnet.features[9],  # Conv4
        )
        with open('model.pkl', 'rb') as f:
            self.classifier = pickle.load(f).cpu()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def expanded_cut(coord, img, view, pad=10):
    x1, y1 = coord[0] - pad, coord[1] - pad
    x2, y2 = coord[0] + view + pad, coord[1] + view + pad
    if x1 < 0:
        x1 += pad
        x2 += pad
    elif x2 > img.shape[0]:
        x1 -= pad
        x2 -= pad
    if y1 < 0:
        y1 += pad
        y2 += pad
    elif y2 > img.shape[1]:
        y1 -= pad
        y2 -= pad
    return img[x1:x2, y1:y2]


def plot_most_activated(activations, images, kernal_size, stride, plot_shape,
                        figsize=(5, 5), image_pad=0, view_pad=0, save_as=None):
    plt.figure(figsize=figsize)
    for i in range(activations.shape[0]):
        index = np.unravel_index(np.argmax(activations[i]),
                                 activations.shape[1:])
        img = (np.rollaxis(images[index[0]], 0, 3) + 1) / 2
        img = np.pad(img,
                     ((image_pad, image_pad), (image_pad, image_pad), (0, 0)),
                     'constant')
        coord = (index[1] * stride, index[2] * stride)
        img = expanded_cut(coord, img, kernal_size, pad=view_pad)
        ax = plt.subplot(plot_shape[0], plot_shape[1], i + 1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.axis('off')
        plt.imshow(img)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    if save_as is not None:
        plt.savefig(save_as)
    plt.show()


def prototypical_images(model, act):
    loss_fn = torch.nn.CrossEntropyLoss()
    mosts = []
    for i, a in enumerate(act):
        images = load_data(a)
        y = Variable(
            torch.from_numpy(np.ones(len(images)) * i).type(dtype_long))
        x = Variable(torch.from_numpy(images).type(dtype_float))
        y_pred = model(x)
        loss = loss_fn(y_pred, y).data.numpy()
        most_a = np.argmin(loss)
        image_save('most_{}.png'.format(a), images[most_a])
        mosts.append(process_image('most_{}.png'.format(a)))
    return mosts


def image_save(filename, img):
    img = (np.rollaxis(img, 0, 3) + 1) / 2
    imsave(filename, img)


def process_image(filename):
    im = imread(filename)[:, :, :3]
    im = im - np.mean(im.flatten())
    im = im / np.max(np.abs(im.flatten()))
    im = np.rollaxis(im, -1).astype(np.float32)
    im = Variable(torch.from_numpy(im).type(dtype_float), requires_grad=True)
    return im


def main():
    act = ['gilpin', 'bracco', 'harmon', 'baldwin', 'hader', 'carell']

    print("Please select part to display output for:\n" +
          "\tFirst Layer \n\tWhole Network")
    question = 'whole network'  # raw_input()

    if question.lower() == "first layer":
        man = MyAlexNet()
        layer1 = man.features[0]
        # Show the weights of the first layer of the neural network
        plt.figure(figsize=(5, 5))
        for i in range(64):
            ax = plt.subplot(8, 8, i + 1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.axis('off')
            img = (np.rollaxis(layer1.weight.data[i].numpy(), 0, 3) + 1) / 2
            plt.imshow(img)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.savefig('layer_1_weights.png')
        plt.show()

        alexnet = torch.nn.Sequential(man.features[0])
        images = []
        for a in act:
            images.append(load_data(a))
        images = Variable(torch.from_numpy(np.vstack(images)).type(dtype_float))
        activations = np.rollaxis(alexnet(images).data.numpy(), 1, 0)
        # Visualize regions of most activations
        plot_most_activated(activations, images.data.numpy(), 11, 4, (8, 8),
                            image_pad=2, save_as='layer1_activate.png')
        # Visualize around regions of most activations
        plot_most_activated(activations, images.data.numpy(), 11, 4, (8, 8),
                            image_pad=2, view_pad=10,
                            save_as='layer1_activate_border.png')

    elif question.lower() == 'whole network':
        print('Loading Pretrained Model')
        model = FaceModel()
        print("\tLoaded Pretrained Model")
        try:
            mosts = []
            for a in act:
                mosts.append(process_image("most_{}.png".format(a)))
        except IOError:
            mosts = prototypical_images(model, act)
        print("\tLoaded Images")

        m = model
        for i, img in enumerate(mosts):
            print("Working on {} of 6".format(i + 1))
            # Vanilla Backpropagation
            model = deepcopy(m)
            VBP = VanillaBackprop(model, img.unsqueeze(0), i)
            vanilla_grads = VBP.generate_gradients()
            save_gradient_images(vanilla_grads, act[i] + '_Vanilla_BP_color')
            grayscale_vanilla_grads = convert_to_grayscale(vanilla_grads)
            save_gradient_images(grayscale_vanilla_grads,
                                 act[i] + '_Vanilla_BP_gray')

            # Guided backpropagation
            model = deepcopy(m)
            GBP = GuidedBackprop(model, img.unsqueeze(0), i)
            guided_grads = GBP.generate_gradients()
            save_gradient_images(guided_grads, act[i] + '_Guided_BP_colour')
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            save_gradient_images(grayscale_guided_grads,
                                 act[i] + '_Guided_BP_gray')
            pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
            save_gradient_images(pos_sal, act[i] + '_pos_sal')
            save_gradient_images(neg_sal, act[i] + '_neg_sal')

            # GradCAM and Guided GradCAM
            orig_im = cv2.imread('most_{}.png'.format(act[i]))
            for j in range(10):
                model = deepcopy(m)
                grad_cam = GradCam(model, target_layer=j)
                cam = grad_cam.generate_cam(img.unsqueeze(0), 0)
                save_class_activation_on_image(orig_im, cam,
                                               act[i] + '_layer' + str(j))
                cam_gb = guided_grad_cam(cam, guided_grads)
                save_gradient_images(cam_gb,
                                     act[i] + '_layer' + str(j) + '_GGrad_Cam')
                grayscale_cam_gb = convert_to_grayscale(cam_gb)
                save_gradient_images(grayscale_cam_gb,
                                     act[i] + '_layer' +
                                     str(j) + '_GGrad_Cam_gray')

            # Layer Visualization
            for j in range(10):
                model = deepcopy(m)
                inverted_representation = InvertedRepresentation(model)
                try:
                    inverted_representation.generate_inverted_image_specific_layer(
                        img.unsqueeze(0), 277, act[i], j)
                except:
                    continue

            # Class Specific Samples
            model = deepcopy(m)
            csig = ClassSpecificImageGeneration(model, i)
            csig.generate(act[i])


if __name__ == '__main__':
    main()
