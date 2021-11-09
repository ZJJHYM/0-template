import torch
import torch.nn as nn


class AsymmetricLossOptimized(nn.Module):
    """Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """ "
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


def cross_entropy_loss(y_true, y_pred):
    #             ce_loss = nn.BCEWithLogitsLoss()
    #             loss = ce_loss(y_pred, y_true)

    #             return loss
    epsilon = 1e-8

    loss = y_true * torch.log(y_pred + epsilon) + (1.0 - y_true) * torch.log(
        1.0 - y_pred + epsilon
    )
    loss = -loss

    return torch.mean(torch.sum(loss, dim=1))


if __name__ == "__main__":
    loss_fn = AsymmetricLossOptimized()

    y_true = torch.ones((32, 82))
    y_pred = torch.rand((32, 82))

    print(loss_fn(y_pred, y_true))
