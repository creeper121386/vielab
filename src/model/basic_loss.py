import torch.nn as nn
import torch


class LTVloss(nn.Module):
    def __init__(self, alpha=1.2, beta=1.5, eps=1e-4):
        super(LTVloss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, origin, illumination, weight):
        '''
        origin:       one batch of input data. shape [batchsize, 3, h, w]
        illumination: one batch of predicted illumination data. if predicted_illumination 
                      is False, then use the output (predicted result) of the network.
        '''

        # # re-normalize origin to 0 ~ 1
        # origin = (input_ - input_.min().item()) / (input_.max().item() - input_.min().item())

        I = origin[:, 0:1, :, :] * 0.299 + origin[:, 1:2, :, :] * \
            0.587 + origin[:, 2:3, :, :] * 0.114
        L = torch.log(I + self.eps)
        dx = L[:, :, :-1, :-1] - L[:, :, :-1, 1:]
        dy = L[:, :, :-1, :-1] - L[:, :, 1:, :-1]

        dx = self.beta / (torch.pow(torch.abs(dx), self.alpha) + self.eps)
        dy = self.beta / (torch.pow(torch.abs(dy), self.alpha) + self.eps)

        x_loss = dx * \
            ((illumination[:, :, :-1, :-1] - illumination[:, :, :-1, 1:]) ** 2)
        y_loss = dy * \
            ((illumination[:, :, :-1, :-1] - illumination[:, :, 1:, :-1]) ** 2)
        tvloss = torch.mean(x_loss + y_loss) / 2.0

        #  print(origin.max().item(), origin.mean().item(), illmination.max().item(), illmination.mean().item(), 'tv loss', tvloss.item(),
        #       I.mean().item(), L.mean().item(), dx.mean().item(), dy.mean().item())
        # exit(0)

        return tvloss * weight
