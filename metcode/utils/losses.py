import torch
import torch.nn as nn
import torch.nn.functional as F



class ContrastiveLoss(nn.Module):
    '''Contrastive loss.
    Takes as inputs the embeddings of two samples and a target label == 1 if samples come 
    from the same class or 0 otherwise.
    Credits to https://github.com/adambielski/siamese-triplet
    '''

    def __init__(self, margin):
        
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.eps = 1e-9


    def forward(self, output1, output2, target, size_average=True):

        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        
        return losses.mean() if size_average else losses.sum()


class CustomLoss(torch.nn.Module):
    def __int__(self):
        super(CustomLoss, self).__init__()

    def forward(self, q1, q2, q3, q4, p1, p2, p3, p4, n1, n2, n3, n4, margin, lam, cos=True):
        if cos:
            score_positive = 1 - F.cosine_similarity(q3, p3)
            score_negative = 1 - F.cosine_similarity(q3, n3)
        else:
            score_positive = F.pairwise_distance(q3, p3, p=2.0)
            score_negative = F.pairwise_distance(q3, n3, p=2.0)
        triplet_loss = torch.mean(
            torch.clamp(torch.pow(score_positive, 2) - torch.pow(score_negative, 2) + margin, min=0.0))
        regular = lam * (F.l1_loss(q1, p1) + F.l1_loss(p2, q2) + F.l1_loss(q4, p4))
        loss = triplet_loss + regular
        return loss


class CustomLoss_vgg(torch.nn.Module):
    def __int__(self):
        super(CustomLoss_vgg, self).__init__()

    def forward(self, q1, q2, q3, q4, q5, q6, p1, p2, p3, p4, p5, p6, n6, margin, lam, cos=True):
        if cos:
            score_positive = 1 - F.cosine_similarity(q6, p6)
            score_negative = 1 - F.cosine_similarity(q6, n6)
        else:
            score_positive = F.pairwise_distance(q6, p6, p=2.0)
            score_negative = F.pairwise_distance(q6, n6, p=2.0)
        triplet_loss = torch.mean(
            torch.clamp(torch.pow(score_positive, 2) - torch.pow(score_negative, 2) + margin, min=0.0))
        d1 = torch.mean(F.pairwise_distance(q1, p1, p=2.0))
        d2 = torch.mean(F.pairwise_distance(q2, p2, p=2.0))
        d3 = torch.mean(F.pairwise_distance(q3, p3, p=2.0))
        d4 = torch.mean(F.pairwise_distance(q4, p4, p=2.0))
        d5 = torch.mean(F.pairwise_distance(q5, p5, p=2.0))
        regular = lam * (d1 + d2 + d3 + d4 + d5)
        # regular = lam * (F.l1_loss(q1, p1) + F.l1_loss(p2, q2) + F.l1_loss(q3, p3) + F.l1_loss(q4, p4) + F.l1_loss(q5, p5))
        loss = triplet_loss + regular
        return loss