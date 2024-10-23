import random
import torch

def batches(data, label, batch_size, use_cuda, randomize=True):
    d_l = list(zip(data, label))
    if randomize:
        random.shuffle(d_l)
    data, label = zip(*d_l)
    data, label = list(data), list(label)

    for i in range(0, len(data), batch_size):
        sentences = data[i:i + batch_size]
        labels = label[i:i + batch_size]

        s_l = zip(sentences, labels)
        s_l = sorted(s_l, key = lambda l: len(l[0]), reverse=True)

        sentences, labels = zip(*s_l)

        sentences = list(sentences)
        labels = list(labels)


        if use_cuda:
            yield [torch.LongTensor(s).cuda() for s in sentences], \
                torch.LongTensor(labels).cuda()
        else:
            yield [torch.LongTensor(s) for s in sentences], \
                torch.LongTensor(labels)