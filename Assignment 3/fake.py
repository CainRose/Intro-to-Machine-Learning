import numpy as np
from collections import Counter
import bayes
import data_processing
import logistic
import torch
import matplotlib.pyplot as plt


def main():
    print("Please select question to display output for:\n" +
          "\t1 2 3 4 5 6 7 8")
    np.random.seed(11)
    torch.manual_seed(11)
    question = '4'  # raw_input()
    fake, real = data_processing.load_data()

    if question == '1':
        print(len(fake), 'fake headlines.')
        print(len(real), 'real headlines.')
        print('Fake headlines average {} words.'
              .format(np.mean([len(w) for w in fake])))
        print('Fake words average {} characters.'
              .format(np.mean([len(c) for w in fake for c in w])))
        print('Real headlines average {} words.'
              .format(np.mean([len(w) for w in real])))
        print('Real words average {} characters.'
              .format(np.mean([len(c) for w in real for c in w])))
        real_count = Counter([w for h in real for w in h])
        real_count_norm = Counter([w for h in real for w in h])
        fake_count = Counter([w for h in fake for w in h])
        fake_count_norm = Counter([w for h in fake for w in h])
        for w in real_count + fake_count:
            if real_count[w] + fake_count[w] < 100:
                real_count_norm[w] = 0
                fake_count_norm[w] = 0
                continue
            real_count_norm[w] = real_count[w] / (real_count[w] + fake_count[w])
            fake_count_norm[w] = 1 - real_count_norm[w]
        real_common = [w[0] for w in real_count_norm.most_common(5)]
        fake_common = [w[0] for w in fake_count_norm.most_common(5)]
        print('The most common words in the real data set, '
              'along with occurences in the real and fake data set')
        for w in real_common:
            print('\t', w, real_count[w], fake_count[w])
        print('The most common words in the fake data set, '
              'along with occurences in the real and fake data set')
        for w in fake_common:
            print('\t', w, real_count[w], fake_count[w])

    elif question == '2':
        data, keywords = data_processing.process_data(fake, real)
        print('Achieved score before tuning the prior: '
              '(Training, Validation, Testing)')
        param = bayes.build_bayes_param(data[0], 1e-20, 1e-20)
        yh = bayes.bayes_is_fake(param, data[0][0])
        print(np.sum(yh == data[0][1]) / len(data[0][1]))
        yh = bayes.bayes_is_fake(param, data[1][0])
        print(np.sum(yh == data[1][1]) / len(data[1][1]))
        yh = bayes.bayes_is_fake(param, data[2][0])
        print(np.sum(yh == data[2][1]) / len(data[2][1]))

        print('Achieved score after tuning the prior: '
              '(Training, Validation, Testing)')
        param, prior = bayes.naive_bayes(data)
        yh = bayes.bayes_is_fake(param, data[0][0])
        print(np.sum(yh == data[0][1]) / len(data[0][1]))
        yh = bayes.bayes_is_fake(param, data[1][0])
        print(np.sum(yh == data[1][1]) / len(data[1][1]))
        yh = bayes.bayes_is_fake(param, data[2][0])
        print(np.sum(yh == data[2][1]) / len(data[2][1]))

    elif question == '3':
        data, keywords = data_processing.process_data(fake, real)
        param, prior = bayes.naive_bayes(data)
        p_fake, p_real = bayes.conditional_probability(
            data[0][0], data[0][1], prior[1], prior[0])

        presence = p_fake / p_real
        absence = (1 - p_fake) / (1 - p_real)

        presence_ind = np.argsort(presence)
        absence_ind = np.argsort(absence)
        presence_real = presence_ind[:40]
        absence_real = absence_ind[:40]
        presence_fake = presence_ind[-40:][::-1]
        absence_fake = absence_ind[-40:][::-1]
        print("Including stop words")
        print("Words whose presence predicts a real headline")
        for i in presence_real[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(1 / presence[i])))
        print("Words whose absence predicts a real headline")
        for i in absence_real[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(1 / absence[i])))
        print("Words whose presence predicts a fake headline")
        for i in presence_fake[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(presence[i])))
        print("Words whose absence predicts a fake headline")
        for i in absence_fake[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(absence[i])))

        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        ENGLISH_STOP_WORDS = list(ENGLISH_STOP_WORDS)
        presence_real = presence_real[np.isin(keywords[presence_real],
                                              ENGLISH_STOP_WORDS, invert=True)]
        absence_real = absence_real[np.isin(keywords[absence_real],
                                            ENGLISH_STOP_WORDS, invert=True)]
        presence_fake = presence_fake[np.isin(keywords[presence_fake],
                                              ENGLISH_STOP_WORDS, invert=True)]
        absence_fake = absence_fake[np.isin(keywords[absence_fake],
                                            ENGLISH_STOP_WORDS, invert=True)]
        print("Excluding stop words")
        print("Words whose presence predicts a real headline")
        for i in presence_real[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(1 / presence[i])))
        print("Words whose absence predicts a real headline")
        for i in absence_real[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(1 / absence[i])))
        print("Words whose presence predicts a fake headline")
        for i in presence_fake[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(presence[i])))
        print("Words whose absence predicts a fake headline")
        for i in absence_fake[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], np.log(absence[i])))

    elif question == '4':
        data, keywords = data_processing.process_data(fake, real, True)

        weight_decay = 0.000061 # logistic.tune_regularization(data)

        model = logistic.build_model(len(keywords)).cuda()
        loss_fn = logistic.cross_entropy_loss
        train_error, valid_error = logistic.train_classifier(
            model, loss_fn, data[0], data[1], iterations=800, batch_size=1000,
            regularization=weight_decay)
        plt.plot(np.arange(0, len(train_error)) * 10, train_error)
        plt.plot(np.arange(0, len(valid_error)) * 10, valid_error)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(('Training Set', 'Validation Set'))
        plt.savefig('q4.png')
        plt.show()

        print('Final Regularization Selected: {:f}'.format(weight_decay))

        actual = data[2][1].data.cpu().numpy()
        prediction = model(data[2][0]).data.cpu().numpy()
        print('Testing Set Error:',
              logistic.get_error(model, loss_fn, data[2])[0])
        print('Testing Set Accuracy:',
              np.mean((prediction > 0.5).T == actual))

        param = model[0].weight.cpu().data.numpy()[0]
        param_sort = np.argsort(param)
        print("Including stop words")
        print("Words whose presence predicts a real headline")
        for i in param_sort[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], param[i]))
        print("Words whose presence predicts a fake headline")
        for i in param_sort[-10:][::-1]:
            print('\t{:15}{:.4f}'.format(keywords[i], param[i]))

        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        ENGLISH_STOP_WORDS = list(ENGLISH_STOP_WORDS)
        param_sort = param_sort[np.isin(keywords[param_sort],
                                              ENGLISH_STOP_WORDS, invert=True)]
        print("Excluding stop words")
        print("Words whose presence predicts a real headline")
        for i in param_sort[:10]:
            print('\t{:15}{:.4f}'.format(keywords[i], param[i]))
        print("Words whose presence predicts a fake headline")
        for i in param_sort[-10:][::-1]:
            print('\t{:15}{:.4f}'.format(keywords[i], param[i]))


if __name__ == '__main__':
    main()
