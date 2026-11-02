"""
FPN (+ RPN + Fast R-CNN head) 학습 step 데모 (더미 데이터).

COCO / VOC 는 무겁기 때문에 더미 이미지와 GT 박스로
  1) ResNet-50 + FPN 으로 P2~P6 계산
  2) 앵커 라벨링(IoU>=0.7 pos, <0.3 neg, GT 별 최고 IoU 앵커도 pos) + 256개 샘플링 -> RPN 손실
  3) RPN proposal + GT 박스로 RoI 구성 -> 128개 샘플링(fg 최대 25%) -> 멀티레벨 RoIAlign -> box head 손실
  4) 두 손실의 합으로 역전파
를 보여준다.

단순화: 이미지 경계를 벗어난 앵커도 그대로 사용(논문은 학습 시 제외), RoI 는 batch 마다 새로 샘플링.

실행:
    python train.py
"""

import torch
import torch.nn.functional as F

from model import FPNDetector, box_iou, decode_boxes, encode_boxes

NUM_CLASSES = 21
BOX_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def make_dummy_batch(n=2, h=256, w=320):
    images = torch.randn(n, 3, h, w)
    boxes = [torch.tensor([[40.0, 30.0, 160.0, 150.0], [180.0, 90.0, 300.0, 230.0]]) for _ in range(n)]
    labels = [torch.tensor([5, 12]) for _ in range(n)]
    return images, boxes, labels


def rpn_loss(obj, reg, anchors, gt_boxes_list, num_samples=256):
    """obj (N,A), reg (N,A,4). pos/neg 를 1:1 까지 샘플링해 BCE + smooth L1."""
    cls_losses, reg_losses = [], []
    for i, gt in enumerate(gt_boxes_list):
        iou = box_iou(gt, anchors)  # (G,A)
        max_iou, argmax = iou.max(dim=0)
        label = torch.full((anchors.size(0),), -1.0, device=anchors.device)  # -1 = ignore
        label[max_iou < 0.3] = 0
        label[max_iou >= 0.7] = 1
        best_per_gt = iou.max(dim=1).values
        label[(iou == best_per_gt[:, None]).any(dim=0)] = 1  # 각 GT 의 최고 IoU 앵커

        pos_idx = (label == 1).nonzero(as_tuple=True)[0]
        neg_idx = (label == 0).nonzero(as_tuple=True)[0]
        pos_idx = pos_idx[torch.randperm(len(pos_idx), device=pos_idx.device)[: num_samples // 2]]
        neg_idx = neg_idx[torch.randperm(len(neg_idx), device=neg_idx.device)[: num_samples - len(pos_idx)]]
        idx = torch.cat([pos_idx, neg_idx])

        cls_losses.append(F.binary_cross_entropy_with_logits(obj[i, idx], label[idx]))
        tgt = encode_boxes(gt[argmax[pos_idx]], anchors[pos_idx])
        reg_losses.append(F.smooth_l1_loss(reg[i, pos_idx], tgt, beta=1 / 9, reduction="sum") / max(len(idx), 1))
    return torch.stack(cls_losses).mean(), torch.stack(reg_losses).mean()


def sample_rois(proposals, gt_boxes_list, gt_labels_list, num_samples=128, fg_frac=0.25):
    """proposal + GT 를 합쳐 fg(IoU>=0.5)/bg 샘플링. rois, labels, regression targets(이미지별) 반환."""
    rois_out, labels_out, targets_out = [], [], []
    for props, gt, gl in zip(proposals, gt_boxes_list, gt_labels_list):
        rois = torch.cat([props, gt])
        iou = box_iou(rois, gt)
        max_iou, argmax = iou.max(dim=1)
        labels = gl[argmax].clone()
        labels[max_iou < 0.5] = 0
        fg = (labels > 0).nonzero(as_tuple=True)[0]
        bg = (labels == 0).nonzero(as_tuple=True)[0]
        fg = fg[torch.randperm(len(fg), device=fg.device)[: int(num_samples * fg_frac)]]
        bg = bg[torch.randperm(len(bg), device=bg.device)[: num_samples - len(fg)]]
        keep = torch.cat([fg, bg])
        rois_out.append(rois[keep])
        labels_out.append(labels[keep])
        targets_out.append(encode_boxes(gt[argmax[keep]], rois[keep], BOX_WEIGHTS))
    return rois_out, labels_out, targets_out


def head_loss(logits, deltas, labels, targets):
    labels, targets = torch.cat(labels), torch.cat(targets)
    cls = F.cross_entropy(logits, labels)
    fg = (labels > 0).nonzero(as_tuple=True)[0]
    d = deltas.view(deltas.size(0), -1, 4)[fg, labels[fg]]  # 정답 클래스의 박스 출력만 사용
    reg = F.smooth_l1_loss(d, targets[fg], beta=1.0, reduction="sum") / max(labels.numel(), 1)
    return cls, reg


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPNDetector(NUM_CLASSES).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)

    model.train()
    for step in range(1, 4):
        images, gt_boxes, gt_labels = make_dummy_batch()
        images = images.to(device)
        gt_boxes = [b.to(device) for b in gt_boxes]
        gt_labels = [l.to(device) for l in gt_labels]
        h, w = images.shape[-2:]

        feats = model.features(images)
        obj, reg, anchors, counts = model.rpn(feats, (h, w))
        l_obj, l_reg = rpn_loss(obj, reg, anchors, gt_boxes)

        proposals = model.propose(obj.detach(), reg.detach(), anchors, counts, (h, w), pre_nms_top_n=500, post_nms_top_n=300)
        rois, labels, targets = sample_rois(proposals, gt_boxes, gt_labels)
        logits, deltas = model.box_forward(feats, rois)
        l_cls, l_box = head_loss(logits, deltas, labels, targets)

        total = l_obj + l_reg + l_cls + l_box
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        print(f"[step {step}] total={total.item():.4f} rpn_obj={l_obj.item():.4f} rpn_reg={l_reg.item():.4f} "
              f"head_cls={l_cls.item():.4f} head_reg={l_box.item():.4f}")


if __name__ == "__main__":
    main()
