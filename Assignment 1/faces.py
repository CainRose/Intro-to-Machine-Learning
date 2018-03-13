from __future__ import absolute_import, division, print_function

import numpy as np
import classifier, get_data
from skimage import io
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


def main():
    print("Please select question to display output for:\n" +
          "\t1 2 3 4 5 6 7 8")
    question = raw_input()

    if question == '1':
        print('The downloading and cropping of images is done by the '
              'get_data function in the get_data.py file.')
        get_data.get_data()
        # For some strange reason, when performing downloading all the images
        # the program hangs after completion, never exiting. 'Done' is printed
        # however. The program does not close even if I forcibly throw an error.
        # On small subsets however, it does close. I suspect it has something to
        # with the automatic garbage collection running into some issue, but I
        # do not have any clue as to what. If you know what could be the cause,
        # please let me know! Otherwise, take this as a notice that when 'Done'
        # is printed the program has finished.
        print('Done')

    elif question == '2':
        print('The loading of images is done by the load_data function in the '
              'get_data.py file. Seperating the images into three sets is done '
              'as part of the classify function in the classifier.py file.')
        images = get_data.load_data('baldwin') # for example
        # Note that by default, the seed is fixed so that repeated identical
        # calls will produce an identical selection of images. This is how we
        # ensure reproducibility. For true randomness, just set the parameter:
        rand_imgs = get_data.load_data('baldwin', is_random=True)

    elif question == '3':
        p, cost, accuracy = classifier.classify(['baldwin', 'carell'])
        print("Training Error:", cost[0])
        print("Validation Error:", cost[1])
        print("Validation Accuracy:", accuracy[1])
        print("Testing Accuracy:", accuracy[2])

    elif question == '4':
        def visualize(p, name):
            vis = p[1:].reshape((32, 32))
            io.imshow(vis)
            plt.savefig(name)
            plt.show()

        print('Full training set')
        p, _, _ = classifier.classify(['baldwin', 'carell'])
        visualize(p, '4a-1.png')
        print('Two image training set')
        p, _, _ = classifier.classify(['baldwin', 'carell'],
                                      set_sizes=(2, 10, 10))
        visualize(p, '4a-2.png')
        print('Stop too early vs stop too late')
        p, _, _ = classifier.classify(['baldwin', 'carell'], max_iter=100)
        visualize(p, '4b-1.png')
        p, _, _ = classifier.classify(['baldwin', 'carell'], epsilon=1e-7)
        visualize(p, '4b-2.png')

    elif question == '5':
        train_act = [['bracco', 'gilpin', 'harmon'],
                     ['baldwin', 'carell', 'hader']]
        other_act = [['chenoweth', 'ferrera', 'drescher'],
                     ['butler', 'vartan', 'radcliffe']]

        # Compare performance against other actors
        def check_performance(name, labels, parameters):
            images = get_data.load_data(name, sizes=[50])
            a = classifier.labelling_accuracy(images, labels, parameters)
            print('\t', name, 'classified with', a, 'accuracy')

        p, _, _ = classifier.classify([train_act[0], train_act[1]],
                                      set_sizes=(66, 10, 10))
        print('Training Actors')
        for act in train_act[0]:
            check_performance(act, np.ones(50), p)
        for act in train_act[1]:
            check_performance(act, np.zeros(50), p)
        print('Other Actors')
        for act in other_act[0]:
            check_performance(act, np.ones(50), p)
        for act in other_act[1]:
            check_performance(act, np.zeros(50), p)

        # Visualize performance vs set size
        plt.axis((0, 70, 0.5, 1))
        plt.xlabel('Size of Training Set')
        plt.ylabel('Classification Accuracy')
        vald = mpatches.Patch(color='blue', label='Validation Set')
        test = mpatches.Patch(color='green', label='Testing Set')
        plt.legend(handles=[vald, test], loc=4)
        plt.ion()
        for i in range(1, 67):
            p, e, a = classifier.classify([train_act[0], train_act[1]],
                                          set_sizes=(i, 10, 10), is_random=True)
            plt.scatter(i, a[1], c='b')
            plt.scatter(i, a[2], c='g')
            plt.pause(0.01)
        plt.savefig('5.png')

    elif question == '6':
        # initialize some data to be used and find gradient
        images = get_data.load_data('bracco', [40])
        param = np.random.random((1025, 6)) * 1e-2
        labels = np.array([[1, 0, 0, 0, 0, 0]] * 40)
        grad = classifier.cost_gradient(images, labels, param)
        h = 1e-6
        # compare against finite differences
        np.random.seed(17)
        for _ in range(5):
            x = np.random.randint(0, 1025)
            y = np.random.randint(0, 6)
            param_mod = param.copy()
            param_mod[x, y] += h
            estimate = (classifier.cost_function(images, labels, param_mod) -
                        classifier.cost_function(images, labels, param)) / h
            print('(p,q) =', (x, y), '-> function:', '{:f}'.format(grad[x, y]),
                  '\t', 'estimate:', '{:f}'.format(estimate))


    elif question == '7':
        act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'carell', 'hader']
        p, cost, accuracy = classifier.classify(act, set_sizes=(66, 10, 10))
        print("Validation Accuracy:", accuracy[1])
        print("Testing Accuracy:", accuracy[2])

    elif question == '8':
        act = ['bracco', 'gilpin', 'harmon', 'baldwin', 'carell', 'hader']
        p, e, a = classifier.classify(act, set_sizes=(66, 10, 10))
        for i, vis in enumerate(p[1:].T):
            vis = vis.reshape((32, 32))
            io.imshow(vis)
            plt.savefig('8-' + str(i) + '.png')
            plt.show()


if __name__ == "__main__":
    main()
