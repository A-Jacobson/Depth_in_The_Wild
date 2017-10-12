import torch
from torch import nn


class RelativeDepthLoss(nn.Module):
    def __init__(self):
        super(RelativeDepthLoss, self).__init__()

    def ranking_loss(self, z_A, z_B, target):
        """
        loss for a given set of pixels:
        z_A: predicted absolute depth for pixels A
        z_B: predicted absolute depth for pixels B
        ground_truth: Relative depth between A and B (-1, 0, 1)
        """
        mask = torch.abs(target)
        predicted_depth = z_A - z_B
        log_loss = torch.log(1 + torch.exp((predicted_depth * target) * -1)) * mask
        squared_loss = (predicted_depth ** 2) * (1-mask)  # if pred depth is not zero adds to loss
        return sum(log_loss + squared_loss)

    def forward(self, output, target):
        total_loss = 0
        for index in range(len(output)):
            # double check index, double check data loader
            x_A = target['x_A'][index].long()
            y_A = target['y_A'][index].long()
            x_B = target['x_B'][index].long()
            y_B = target['y_B'][index].long()

            z_A = output[index][0][x_A, y_A]  # all "A" points
            z_B = output[index][0][x_B, y_B]  # all "B" points

            total_loss += self.ranking_loss(z_A, z_B, target['ordinal_relation'][index])

        return total_loss / len(output)
