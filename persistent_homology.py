import torch
from torch import nn
import gudhi as gd


class differentiable_persistence(torch.autograd.Function):
    # TODO: manage features with threshold instead of a fixed number

    @staticmethod
    def forward(ctx, input, max_features):

        cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1 - input).flatten(), dimensions=input.shape)
        intervals = cubical_complex.persistence()
        intervals = [[x[1] for x in intervals if x[0] == dim] for dim in range(len(input.shape))]
        for i in range(len(intervals)):
            if len(intervals[i]) > max_features:
                intervals[i] = intervals[i][:max_features]
            for _ in range(len(intervals[i]), max_features):
                intervals[i].append((0.0, 0.0))
        intervals = torch.tensor(intervals)
        intervals[0][0][1] = 1.0

        ctx.save_for_backward(input, intervals)

        return intervals

    @staticmethod
    def backward(ctx, grad_output):
        input, intervals, = ctx.saved_tensors

        e = 0.05

        grad = torch.zeros([*grad_output.shape, *input.shape])

        for degree_i in range(grad_output.shape[0]):
            for interval_i in range(grad_output.shape[1]):
                pb, pd = intervals[degree_i][interval_i]
                if pb > e:
                    grad[degree_i, interval_i, 0] = -torch.isclose(input, 1-pb).float()
                if pd < 1 - e:
                    grad[degree_i, interval_i, 1] = -torch.isclose(input, 1-pd).float()

        res = torch.mm(grad_output.view(1, -1), grad.view(torch.prod(torch.tensor(grad_output.shape)), torch.prod(torch.tensor(input.shape))))

        return res.view(input.shape).to(input.device), None


def topo_loss_fn(ph, y):
        # ph.shape = [rank, max_features, 2]
        # y = [b0, b1, ..] rank elements

        dims, max_features, _ = ph.shape

        considered_dims = min(dims, len(y))

        # TODO: add assert


        loss = torch.tensor(0.0)
        for rank in range(considered_dims):
            for i in range(y[rank]):
                loss += 1.0 - torch.pow((ph[rank][i][1] - ph[rank][i][0]), 2)
            for i in range(y[rank], max_features):
                loss += torch.pow((ph[rank][i][1] - ph[rank][i][0]), 2)

        return torch.sqrt(loss/(considered_dims*max_features))


class TopoLossMIP():
    def __init__(self, max_features, channel):
        # expected input [batch, channel, H, W, D]
        self.ph = PH(max_features=max_features)
        self.c = channel
    def __call__(self, batch, y):
        running_loss = torch.tensor(0.0)
        for i in range(len(batch)):
            pred = batch[i]
            intervals = self.ph(pred[self.c].max(2).values)
            running_loss += topo_loss_fn(intervals, y[i])
        return running_loss/len(batch)


class DiceTopoLoss():
    def __init__(self, dice, topo, eta):

        self.dice = dice
        self.topo = topo

        self.eta = eta

    def __call__(self, pred, y):

        dice_loss = self.dice(pred, y[0])
        topo_loss = self.topo(nn.Sigmoid()(pred), y[1])

        return self.eta*dice_loss + (1-self.eta)*topo_loss


class PH(nn.Module):
    def __init__(self, max_features):
        super().__init__()

        self.max_features = max_features

    def forward(self, x):
        return differentiable_persistence.apply(x, self.max_features)


def compute_betti(bin_img):
    cubical_complex = gd.CubicalComplex(top_dimensional_cells=(1 - bin_img).flatten(), dimensions=bin_img.shape)
    cubical_complex.compute_persistence()
    return cubical_complex.persistent_betti_numbers(0.5, 0.6)[:-1]


class image3dTopo():
    def __init__(self, max_features):

        self.pg = PH(max_features)

    def __cal__(self, x):
        # x.shape = [B, 1, H, W, D]
        pass