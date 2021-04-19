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


# zero dce loss:
class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

        return k


class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

        self.DEVICE_CORRECT = False

    def set_device(self, device):
        self.weight_left = self.weight_left.to(device)
        self.weight_right = self.weight_right.to(device)
        self.weight_up = self.weight_up.to(device)
        self.weight_down = self.weight_down.to(device)
        self.DEVICE_CORRECT = True

    def forward(self, org, enhance):
        b, c, h, w = org.shape
        device = org.device

        if not self.DEVICE_CORRECT:
            self.set_device(device)

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).to(device) + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).to(device),
                                                                  torch.FloatTensor([0]).to(device)),
            torch.FloatTensor([0.5]).to(device))
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(device)), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E


class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
