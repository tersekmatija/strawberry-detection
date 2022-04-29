import torch
import torch.nn as nn
import math
import numpy as np

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

class DetectionLoss(nn.Module):

    def __init__(self, nc, gr, anchors = (), train_anchor_threshold = 4.0, balance = [4.0, 1.0, 0.4], stride = [8, 16, 32]):
        super(DetectionLoss, self).__init__()
        self.train_anchor_threshold = train_anchor_threshold
        self.na = len(anchors[0]) // 2
        self.nl = len(anchors)
        print("NL"+str(self.nl))
        self.gr = gr
        self.nc = nc

        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.Tensor(stride).float().view(-1, 1, 1)
        self.balance = balance

        self.bce = nn.BCEWithLogitsLoss()

    # calculate detection loss
    def forward(self, predictions, targets):
        device = targets[0].device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(predictions, targets)  # targets

        cp, cn = smooth_BCE(eps=0.0)

        nt = 0  # number of targets
        no = len(predictions)  # number of output

        #print(no)
        for i, pi in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += self.bce(pi[..., 4], tobj) * self.balance[i]  # obj loss
        return lbox, lobj, lcls

    def build_targets(self, predictions, targets):

        self.anchors = self.anchors.to(targets.device)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)

        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
        
        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets
        
        for i in range(self.nl):
            anchors = self.anchors[i] #[3,2]
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            # Match targets to anchors
            t = targets * gain

            if nt:
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.train_anchor_threshold  # compare

                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

class SegmentationLoss(nn.Module):

    def __init__(self, num_classes):
        super(SegmentationLoss, self).__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.nc = num_classes

    def forward(self, predictions, targets):
        ps = predictions.view(-1)
        ts = targets.view(-1)
        self.bce = self.bce.to(ps.device)
        lseg = self.bce(ps, ts)
        
        preds = torch.argmax(predictions, dim = 1)
        preds = torch.unsqueeze(preds, 1)

        targets = torch.argmax(targets, dim = 1)
        masks = torch.unsqueeze(targets, 1)
      
        ious = torch.zeros(preds.shape[0])
        present_classes = torch.zeros(preds.shape[0])
        for cls in range(self.nc):
            masks_c = masks == cls
            outputs_c = preds == cls
            TP = torch.sum(torch.logical_and(masks_c, outputs_c), dim = [1, 2, 3]).cpu()
            FP = torch.sum(torch.logical_and(torch.logical_not(masks_c), outputs_c), dim = [1, 2, 3]).cpu()
            FN = torch.sum(torch.logical_and(masks_c, torch.logical_not(outputs_c)), dim = [1, 2, 3]).cpu()
            ious += torch.nan_to_num(TP / (TP + FP + FN))
            present_classes += (masks.view(preds.shape[0], -1) == cls).any(dim = 1).cpu()

        iou = torch.mean(ious / present_classes)

        liou = 1 - iou

        return lseg, liou


class CombinedLoss(nn.Module):

    def __init__(self, nc, gr, anchors = (), train_anchor_threshold = 4.0, balance = [4.0, 1.0, 0.4], stride=[8, 16, 32]):
        super(CombinedLoss, self).__init__()
        self.train_anchor_threshold = train_anchor_threshold
        self.na = len(anchors[0]) // 2
        self.nl = len(anchors)
        self.gr = gr
        self.nc = nc
        self.anchors = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.balance = balance
        self.stride = stride

        self.seg_loss = SegmentationLoss(nc)
        self.det_loss = DetectionLoss(nc, gr, anchors=anchors, stride=stride)

    def forward(self, predictions, targets):

        lseg, liou = self.seg_loss(predictions[0], targets[0])
        lbox, lobj, lcls = self.det_loss(predictions[1], targets[1])

        return lseg, liou, lbox, lobj, lcls