import torch
import torch.nn as nn
import torch.nn.functional as F
from .nn_distance.chamfer_loss import ChamferLoss


class Loss(nn.Module):
    """ Loss for training DeformNet.
        Use NOCS coords to supervise training.
    """
    def __init__(self, corr_wt, cd_wt, entropy_wt, deform_wt):
        super(Loss, self).__init__()
        self.threshold = 0.1
        self.chamferloss = ChamferLoss()
        self.corr_wt = corr_wt
        self.cd_wt = cd_wt
        self.entropy_wt = entropy_wt
        self.deform_wt = deform_wt

    def forward(self, assign_mat, deltas, prior, nocs, model):
        """
        Args:
            assign_mat: bs x n_pts x nv
            deltas: bs x nv x 3
            prior: bs x nv x 3
        """
        inst_shape = prior + deltas
        # smooth L1 loss for correspondences
        soft_assign = F.softmax(assign_mat, dim=2)
        coords = torch.bmm(soft_assign, inst_shape)  # bs x n_pts x 3
        diff = torch.abs(coords - nocs)
        less = torch.pow(diff, 2) / (2.0 * self.threshold)
        higher = diff - self.threshold / 2.0
        corr_loss = torch.where(diff > self.threshold, higher, less)
        corr_loss = torch.mean(torch.sum(corr_loss, dim=2))
        corr_loss = self.corr_wt * corr_loss
        # entropy loss to encourage peaked distribution
        log_assign = F.log_softmax(assign_mat, dim=2)
        entropy_loss = torch.mean(-torch.sum(soft_assign * log_assign, 2))
        entropy_loss = self.entropy_wt * entropy_loss
        # cd-loss for instance reconstruction
        cd_loss, _, _ = self.chamferloss(inst_shape, model)
        cd_loss = self.cd_wt * cd_loss
        # L2 regularizations on deformation
        deform_loss = torch.norm(deltas, p=2, dim=2).mean()
        deform_loss = self.deform_wt * deform_loss
        # total loss
        total_loss = corr_loss + entropy_loss + cd_loss + deform_loss
        return total_loss, corr_loss, cd_loss, entropy_loss, deform_loss
