import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class ConditionalBatchNorm2d(nn.BatchNorm2d):

    """Conditional Batch Normalization"""

    def __init__(self, num_features, eps=1e-05, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(ConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )

    def forward(self, input, weight, bias, **kwargs):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(input, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)
        size = output.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * output + bias


class CategoricalConditionalBatchNorm2d(ConditionalBatchNorm2d):

    def __init__(self, num_classes, num_features, eps=1e-5, momentum=0.1,
                 affine=False, track_running_stats=True):
        super(CategoricalConditionalBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.weights = nn.Embedding(num_classes, num_features)
        self.biases = nn.Embedding(num_classes, num_features)

        self._initialize()

    def _initialize(self):
        init.ones_(self.weights.weight.data)
        init.zeros_(self.biases.weight.data)

    def forward(self, input, c, **kwargs):
        weight = self.weights(c)
        bias = self.biases(c)

        return super(CategoricalConditionalBatchNorm2d, self).forward(input, weight, bias)


if __name__ == '__main__':
    """Forward computation check."""
    import torch
    size = (3, 3, 12, 12)
    batch_size, num_features = size[:2]
    print('# Affirm embedding output')
    naive_bn = nn.BatchNorm2d(3)
    idx_input = torch.tensor([1, 2, 0], dtype=torch.long)
    embedding = nn.Embedding(3, 3)
    weights = embedding(idx_input)
    print('# weights size', weights.size())
    empty = torch.tensor((), dtype=torch.float)
    running_mean = empty.new_zeros((3,))
    running_var = empty.new_ones((3,))

    naive_bn_W = naive_bn.weight
    # print('# weights from embedding | type {}\n'.format(type(weights)), weights)
    # print('# naive_bn_W | type {}\n'.format(type(naive_bn_W)), naive_bn_W)
    input = torch.rand(*size, dtype=torch.float32)
    print('input size', input.size())
    print('input ndim ', input.dim())

    _ = naive_bn(input)

    print('# batch_norm with given weights')

    try:
        with torch.no_grad():
            output = F.batch_norm(input, running_mean, running_var,
                                  weights, naive_bn.bias, False, 0.0, 1e-05)
    except Exception as e:
        print("\tFailed to use given weights")
        print('# Error msg:', e)
        print()
    else:
        print("Succeeded to use given weights")

    print('\n# Batch norm before use given weights')
    with torch.no_grad():
        tmp_out = F.batch_norm(input, running_mean, running_var,
                               naive_bn_W, naive_bn.bias, False, .0, 1e-05)
    weights_cast = weights.unsqueeze(-1).unsqueeze(-1)
    weights_cast = weights_cast.expand(tmp_out.size())
    try:
        out = weights_cast * tmp_out
    except Exception:
        print("Failed")
    else:
        print("Succeeded!")
        print('\t {}'.format(out.size()))
        print(type(tuple(out.size())))

    print('--- condBN and catCondBN ---')

    catCondBN = CategoricalConditionalBatchNorm2d(3, 3)
    output = catCondBN(input, idx_input)

    assert tuple(output.size()) == size

    condBN = ConditionalBatchNorm2d(3)

    idx = torch.tensor([1], dtype=torch.long)
    out = catCondBN(input, idx)

    print('cat cond BN weights\n', catCondBN.weights.weight.data)
    print('cat cond BN biases\n', catCondBN.biases.weight.data)
