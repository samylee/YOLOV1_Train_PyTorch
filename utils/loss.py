import torch
import torch.nn as nn

class YOLOV1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20, coord_scale=5, noobject_scale=0.5, device=torch.device('cuda:0')):
        super(YOLOV1Loss, self).__init__()
        self.S, self.B, self.C = S, B, C
        self.coord_scale, self.noobject_scale = coord_scale, noobject_scale
        self.device = device
        self.coords = 4

        self.class_criterion = nn.MSELoss(reduction='sum')
        self.noobj_criterion = nn.MSELoss(reduction='sum')
        self.xy_criterion = nn.MSELoss(reduction='sum')
        self.wh_criterion = nn.MSELoss(reduction='sum')
        self.obj_criterion = nn.MSELoss(reduction='sum')

    def box_iou(self, box1, box2):
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])

        area1 = box_area(box1.T)
        area2 = box_area(box2.T)

        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        # iou = inter / (area1 + area2 - inter)
        return inter / (area1[:, None] + area2 - inter)

    def box_rmse(self, box1, box2):
        return torch.sqrt(
            torch.pow(box1[:, 0] - box2[:, 0], 2) +
            torch.pow(box1[:, 1] - box2[:, 1], 2) +
            torch.pow(box1[:, 2] - box2[:, 2], 2) +
            torch.pow(box1[:, 3] - box2[:, 3], 2)
        )

    def forward(self, preds, targets):
        batch_size = preds.size(0)

        pobj_index = self.S * self.S * self.C
        pbox_index = self.S * self.S * (self.C + self.B)
        preds_class = preds[:, :pobj_index].view(-1, self.S * self.S, self.C)
        preds_obj = preds[:, pobj_index:pbox_index].view(-1, self.S * self.S, self.B)
        preds_coord = preds[:, pbox_index:].view(-1, self.S * self.S, self.B * self.coords)

        tobj_index = self.S * self.S * self.C
        tbox_index = self.S * self.S * (self.C + 1)
        targets_class = targets[:, :tobj_index].view(-1, self.S * self.S, self.C)
        targets_obj = targets[:, tobj_index:tbox_index].view(-1, self.S * self.S, 1)
        targets_coord = targets[:, tbox_index:].view(-1, self.S * self.S, self.coords)

        # 1. classes loss
        class_mask = (targets_obj > 0).expand_as(preds_class)

        preds_class_mask = preds_class[class_mask]
        targets_class_mask = targets_class[class_mask]

        loss_class = self.class_criterion(preds_class_mask, targets_class_mask)

        # 2. noobj loss
        noobj_mask = (targets_obj == 0).expand_as(preds_obj)

        preds_noobj_mask = preds_obj[noobj_mask]

        loss_noobj = self.noobj_criterion(preds_noobj_mask, torch.zeros_like(preds_noobj_mask))

        # 3. coord loss
        pcoord_mask = (targets_obj > 0).expand_as(preds_coord)
        tcoord_mask = (targets_obj > 0).expand_as(targets_coord)

        preds_coord_mask = preds_coord[pcoord_mask].view(-1, self.coords)
        targets_coord_mask = targets_coord[tcoord_mask].view(-1, self.coords)

        best_index_mask = torch.empty((preds_coord_mask.size(0), 1), dtype=torch.bool).to(self.device).fill_(False)
        worst_index_mask = torch.empty((preds_coord_mask.size(0), 1), dtype=torch.bool).to(self.device).fill_(True)
        best_iou_mask = torch.empty((targets_coord_mask.size(0), 1), dtype=torch.float32).to(self.device)

        for i, j in zip(range(0, preds_coord_mask.size(0), self.B), range(targets_coord_mask.size(0))):
            bbox_pred = preds_coord_mask[i:i+self.B]
            pred_xyxy = torch.empty(bbox_pred.size(), dtype=torch.float32).to(self.device)
            pred_xyxy[:, :2] = bbox_pred[:, :2] / float(self.S) - 0.5 * bbox_pred[:, 2:4] * bbox_pred[:, 2:4]
            pred_xyxy[:, 2:4] = bbox_pred[:, :2] / float(self.S) + 0.5 * bbox_pred[:, 2:4] * bbox_pred[:, 2:4]

            bbox_target = targets_coord_mask[j].unsqueeze(0)
            target_xyxy = torch.empty(bbox_target.size(), dtype=torch.float32).to(self.device)
            target_xyxy[:, :2] = bbox_target[:, :2] / float(self.S) - 0.5 * bbox_target[:, 2:4]
            target_xyxy[:, 2:4] = bbox_target[:, :2] / float(self.S) + 0.5 * bbox_target[:, 2:4]

            iou = self.box_iou(pred_xyxy, target_xyxy)
            max_iou, max_index = iou.max(0)
            if max_iou == 0:
                rmse = self.box_rmse(pred_xyxy, target_xyxy)
                _, max_index = rmse.min(0)

            best_index_mask[i + max_index] = True
            worst_index_mask[i + max_index] = False
            best_iou_mask[j] = max_iou

        best_preds_coord = preds_coord_mask[best_index_mask.expand_as(preds_coord_mask)].view(-1, self.coords)
        loss_xy = self.xy_criterion(best_preds_coord[:, :2], targets_coord_mask[:, :2])
        loss_wh = self.wh_criterion(best_preds_coord[:, 2:], torch.sqrt(targets_coord_mask[:, 2:]))

        # 4. obj loss and noobj_append loss
        pobj_mask = (targets_obj > 0).expand_as(preds_obj)
        preds_obj_mask = preds_obj[pobj_mask].unsqueeze(-1)

        best_preds_obj = preds_obj_mask[best_index_mask]
        worst_preds_obj = preds_obj_mask[worst_index_mask]

        loss_obj = self.obj_criterion(best_preds_obj, best_iou_mask.squeeze(-1))
        loss_noobj_append = self.noobj_criterion(worst_preds_obj, torch.zeros_like(worst_preds_obj))

        # 5. total loss
        loss = self.coord_scale * (loss_xy + loss_wh) + loss_obj + self.noobject_scale * (loss_noobj + loss_noobj_append) + loss_class

        return loss/batch_size
