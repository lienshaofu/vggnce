import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np


class NEG_loss(nn.Module):
    def __init__(self, num_classes, embed_size, weights=None):
        """
        :param num_classes: An int. The number of possible classes.
        :param embed_size: An int. EmbeddingLockup size
        :param num_sampled: An int. The number of sampled from noise examples
        :param weights: A list of non negative floats. Class weights. None if
            using uniform sampling. The weights are calculated prior to
            estimation and can be of any form, e.g equation (5) in [1]
        """

        super(NEG_loss, self).__init__()

        self.num_classes = num_classes
        self.embed_size = embed_size

        self.out_embed = nn.Embedding(self.num_classes, self.embed_size, sparse=True)
        self.out_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

        self.in_embed = nn.Embedding(self.num_classes, self.embed_size, sparse=True)
        self.in_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-1, 1))

        self.weights = weights
        if self.weights is not None:
            assert min(self.weights) >= 0, "Each weight should be >= 0"

            self.weights = Variable(t.from_numpy(weights)).float()

    def sample(self, num_sample):
        """
        draws a sample from classes based on weights
        """
        return t.multinomial(self.weights, num_sample, True)

    def forward(self, input_labes, out_labels, num_sampled):
        """
        :param input_labes: Tensor with shape of [batch_size] of Long type
        :param out_labels: Tensor with shape of [batch_size, window_size] of Long type
        :param num_sampled: An int. The number of sampled from noise examples
        :return: Loss estimation with shape of [1]
            loss defined in Mikolov et al. Distributed Representations of Words and Phrases and their Compositionality
            papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
        """

        use_cuda = self.out_embed.weight.is_cuda
        print (out_labels.size())

        [batch_size, window_size] = out_labels.size()


        input = self.in_embed(input_labes.repeat(1, window_size).contiguous().view(-1))
        output = self.out_embed(out_labels.contiguous().view(-1))

        if self.weights is not None:
            noise_sample_count = batch_size * window_size * num_sampled
            draw = self.sample(noise_sample_count)
            noise = draw.view(batch_size * window_size, num_sampled)
        else:
            noise = Variable(t.Tensor(batch_size * window_size, num_sampled).
                             uniform_(0, self.num_classes - 1).long())

        if use_cuda:
            noise = noise.cuda()
        noise = self.out_embed(noise).neg()

        log_target = (input * output).sum(1).squeeze().sigmoid().log()

        sum_log_sampled = t.bmm(noise, input.unsqueeze(2)).sigmoid().log().sum(1).squeeze()

        loss = log_target + sum_log_sampled

        return -loss.sum() / batch_size

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()