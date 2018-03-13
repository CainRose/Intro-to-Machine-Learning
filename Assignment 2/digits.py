from __future__ import absolute_import, division, print_function

import neural_network as nn
import numpy as np
import cPickle
from matplotlib import pyplot as plt


def main():
    print("Please select question to display output for:\n" +
          "\t1 2 3 4 5 6")
    question = raw_input()

    from scipy.io import loadmat
    M = loadmat("mnist_all.mat")
    np.random.seed(10007)  # Ensure consistent results

    if question == '1':
        print("Example images used in the report are being displayed.")
        for i in range(10):
            m = np.vstack((M['train' + str(i)], M['test' + str(i)]))
            digits = m[np.random.randint(0, len(m), 10)]
            plt.figure(i + 1, figsize=(10, 4))
            for j in range(10):
                plt.subplot(2, 5, j + 1)
                plt.imshow(digits[j].reshape((28, 28)), cmap=plt.cm.gray)
            plt.tight_layout()
            plt.savefig('sample_digits_' + str(i) + '.png')
        plt.show()

    elif question == '2':
        print("Nothing to display; all content presented fully in the report.")

    elif question == '3':
        print("Displaying correctness of gradient descent.")
        h = 1e-5
        print("Correctness for changes in weights.")
        for i in range(10):
            y = np.random.randint(10)
            m = M['train' + str(y)] / 255.0
            y = [int(j == y) for j in range(10)]
            y = np.tile(y, (len(m), 1))
            W = np.random.normal(0, 1 / 140, (28 * 28, 10))
            b = np.random.normal(0, 1 / 140, 10)
            dW, db = nn.basic_gradient(W, m, b, y)

            x_c = np.random.randint(W.shape[0])
            y_c = np.random.randint(W.shape[1])
            W_mod = W.copy()
            W_mod[x_c, y_c] += h
            dW_est = (nn.NLL(nn.basic_network(W_mod, m, b), y) -
                      nn.NLL(nn.basic_network(W, m, b), y)) / h

            print("Random point {} (weights). Computed, Estimate: {:f} {:f}"
                  "".format(i, dW[x_c, y_c], dW_est))
        print("Correctness for changes in biases.")
        for i in range(10):
            y = np.random.randint(10)
            m = M['train' + str(y)] / 255.0
            y = [int(j == y) for j in range(10)]
            y = np.tile(y, (len(m), 1))
            W = np.random.normal(0, 1 / 140, (28 * 28, 10))
            b = np.random.normal(0, 1 / 140, 10)
            dW, db = nn.basic_gradient(W, m, b, y)

            x_c = np.random.randint(b.shape[0])
            b_mod = b.copy()
            b_mod[x_c] += h
            db_est = (nn.NLL(nn.basic_network(W, m, b_mod), y) -
                      nn.NLL(nn.basic_network(W, m, b), y)) / h

            print("Random point {} (biases). Computed, Estimate: {:f} {:f}"
                  "".format(i, db[x_c], db_est))

    elif question == '4':
        print("Starting basic gradient descent.")
        W = np.random.normal(0, 1 / 140, (28 * 28, 10))
        b = np.random.normal(0, 1 / 140, 10)

        l = np.array([1] + [0] * 9)
        x = np.vstack([M['train' + str(i)] for i in range(10)]) / 255
        y = np.vstack([np.tile(np.roll(l, i), (M['train' + str(i)].shape[0], 1))
                       for i in range(10)])
        xv = np.vstack([M['test' + str(i)] for i in range(10)]) / 255
        yv = np.vstack([np.tile(np.roll(l, i), (M['test' + str(i)].shape[0], 1))
                        for i in range(10)])

        W, b = nn.gradient_descent(W, x, b, y, xv, yv, momentum=0,
                                   save_file='learning_rate.png')
        plt.figure(figsize=(10, 4))
        for i, x in enumerate(W.T):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
        plt.tight_layout()
        plt.savefig('digit_visualization.png')
        plt.show()

    elif question == '5':
        print("Starting gradient descent with momentum.")
        W = np.random.normal(0, 1 / 140, (28 * 28, 10))
        b = np.random.normal(0, 1 / 140, 10)

        l = np.array([1] + [0] * 9)
        x = np.vstack([M['train' + str(i)] for i in range(10)]) / 255
        y = np.vstack([np.tile(np.roll(l, i), (M['train' + str(i)].shape[0], 1))
                       for i in range(10)])
        xv = np.vstack([M['test' + str(i)] for i in range(10)]) / 255
        yv = np.vstack([np.tile(np.roll(l, i), (M['test' + str(i)].shape[0], 1))
                        for i in range(10)])

        W, b = nn.gradient_descent(W, x, b, y, xv, yv,
                                   save_file='learning_rate_momentum.png')
        plt.figure(figsize=(10, 4))
        for i, x in enumerate(W.T):
            plt.subplot(2, 5, i + 1)
            plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
        plt.tight_layout()
        plt.savefig('digit_visualization_momentum.png')
        plt.show()

        with open('nn.pkl', 'wb') as f:
            cPickle.dump((W, b), f)

    elif question == '6':
        try:
            with open('nn.pkl', 'rb') as f:
                W, b = cPickle.load(f)
        except IOError:
            print('Trained neural network does not exists. '
                  'Please run question 5 to train and store network.')
            return

        l = np.array([1] + [0] * 9)
        x = np.vstack([M['train' + str(i)] for i in range(10)]) / 255
        y = np.vstack([np.tile(np.roll(l, i), (M['train' + str(i)].shape[0], 1))
                       for i in range(10)])

        size = 0.4
        w, z = 13, 13
        n1, n2 = 0, 1
        w1o, w2o = W[w * 28 + z, n1], W[w * 28 + z, n2]
        w1v = np.linspace(w1o - size, w1o + size, 10)
        w2v = np.linspace(w2o - size, w2o + size, 10)
        meshw, meshz = np.meshgrid(w1v, w2v)

        try:
            with open('C.pkl', 'rb') as f:
                C = cPickle.load(f)
        except IOError:
            C = np.empty((w1v.size, w2v.size))
            for i, w1 in enumerate(w1v):
                for j, w2 in enumerate(w2v):
                    print(i, j)
                    W[w * 28 + z, n1] = w1
                    W[w * 28 + z, n2] = w2
                    C[i, j] = nn.NLL(nn.basic_network(W, x, b), y)
            with open('C.pkl', 'wb') as f:
                cPickle.dump(C, f)

        plt.contour(meshw, meshz, C.T, cmap=plt.cm.gray)
        plt.xlabel('$w_1$')
        plt.ylabel('$w_2$')

        try:
            with open('w_paths.pkl', 'rb') as f:
                w1, w2, w1m, w2m = cPickle.load(f)
        except IOError:
            w1 = [w1o - 0.3]
            w2 = [w2o - 0.3]
            for i in range(10):
                print(i)
                W[w * 28 + z, n1] = w1[-1]
                W[w * 28 + z, n2] = w2[-1]
                dW, _ = nn.basic_gradient(W, x, b, y)
                w1.append(w1[-1] - dW[w * 28 + z, n1] * 6e-3)
                w2.append(w2[-1] - dW[w * 28 + z, n2] * 6e-3)

            w1m = [w1o - 0.3]
            w2m = [w2o - 0.3]
            dW = np.zeros_like(W)
            for i in range(10):
                print(i)
                W[w * 28 + z, n1] = w1m[-1]
                W[w * 28 + z, n2] = w2m[-1]
                dW_n, _ = nn.basic_gradient(W, x, b, y)
                dW = dW * 0.25 - dW_n * 6e-3
                w1m.append(w1m[-1] + dW[w * 28 + z, n1])
                w2m.append(w2m[-1] + dW[w * 28 + z, n2])
            with open('w_paths.pkl', 'wb') as f:
                cPickle.dump((w1, w2, w1m, w2m), f)

        plt.plot(w1, w2, 'bo-')
        plt.plot(w1m, w2m, 'ro-')
        plt.legend(('No Momentum', 'Momentum'))
        plt.savefig('mom_vs_no_mom.png')
        plt.show()

        plt.figure()
        plt.contour(meshw, meshz, C.T, cmap=plt.cm.gray)
        plt.xlabel('$w_1$')
        plt.ylabel('$w_2$')

        try:
            with open('w_paths_2.pkl', 'rb') as f:
                w1, w2, w1m, w2m = cPickle.load(f)
        except IOError:
            w1 = [w1o - 0.3]
            w2 = [w2o - 0.3]
            for i in range(10):
                print(i)
                W[w * 28 + z, n1] = w1[-1]
                W[w * 28 + z, n2] = w2[-1]
                dW, _ = nn.basic_gradient(W, x, b, y)
                w1.append(w1[-1] - dW[w * 28 + z, n1] * 6e-3)
                w2.append(w2[-1] - dW[w * 28 + z, n2] * 6e-3)

            w1m = [w1o - 0.3]
            w2m = [w2o - 0.3]
            dW = np.zeros_like(W)
            for i in range(10):
                print(i)
                W[w * 28 + z, n1] = w1m[-1]
                W[w * 28 + z, n2] = w2m[-1]
                dW_n, _ = nn.basic_gradient(W, x, b, y)
                dW = dW * 0.8 - dW_n * 6e-3
                w1m.append(w1m[-1] + dW[w * 28 + z, n1])
                w2m.append(w2m[-1] + dW[w * 28 + z, n2])
            with open('w_paths_2.pkl', 'wb') as f:
                cPickle.dump((w1, w2, w1m, w2m), f)

        plt.plot(w1, w2, 'bo-')
        plt.plot(w1m, w2m, 'ro-')
        plt.legend(('No Momentum', 'Momentum'))
        plt.savefig('mom_vs_no_mom_2.png')
        plt.show()


    else:
        print("Invalid question selected. Please try again.")


if __name__ == "__main__":
    main()
