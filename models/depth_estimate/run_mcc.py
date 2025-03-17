import torch
import cv2
import numpy as np
from tqdm import tqdm

from ultils.weight import download_MCC_model
import ultils.mcc.misc as misc 
from models.depth_estimate.MCC import MCC_model, MCC_engine
from pytorch3d.io.obj_io import load_obj
from pytorch3d.structures import Pointclouds

def pad_image(im, value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + value)], dim=0)


def normalize(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


class run_MCC():
    def __init__(self, args):
        model_path = download_MCC_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        args.resume = model_path
        args.viz_granularity = args.granularity
        self.model = MCC_model.get_mcc_model(
            occupancy_weight=1.0,
            rgb_weight=0.01,
            args=args,
        ).cuda()
        misc.load_model(args=args, model_without_ddp=self.model, optimizer=None, loss_scaler=None)
        # print(self.model.load_state_dict(checkpoint["model"], strict=False))
        self.model.eval() 
    def predict(self, args):
        score_thresholds=[0.9, 0.7, 0.5]
        rgb = cv2.imread(args.image)
        obj = load_obj(args.point_cloud)
        seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
        H, W = seen_rgb.shape[:2]
        seen_rgb = torch.nn.functional.interpolate(
            seen_rgb.permute(2, 0, 1)[None],
            size=[H, W],
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)

        seen_xyz = obj[0].reshape(H, W, 3)
        seg = cv2.imread(args.seg, cv2.IMREAD_UNCHANGED)
        mask = torch.tensor(cv2.resize(seg, (W, H))).bool()
        mask = mask[:, :, :3]  # Giữ lại 3 kênh đầu tiên
        seen_xyz[~mask] = float('inf')

        seen_xyz = normalize(seen_xyz)

        # bottom, right = mask.nonzero().max(dim=0)[0]
        # top, left = mask.nonzero().min(dim=0)[0]
        bottom, right = mask[..., 0].nonzero().max(dim=0)[0]
        top, left = mask[..., 0].nonzero().min(dim=0)[0]
        bottom = bottom + 40
        right = right + 40
        top = max(top - 40, 0)
        left = max(left - 40, 0)

        seen_xyz = seen_xyz[top:bottom+1, left:right+1]
        seen_rgb = seen_rgb[top:bottom+1, left:right+1]

        seen_xyz = pad_image(seen_xyz, float('inf'))
        seen_rgb = pad_image(seen_rgb, 0)

        seen_rgb = torch.nn.functional.interpolate(
            seen_rgb.permute(2, 0, 1)[None],
            size=[800, 800],
            mode="bilinear",
            align_corners=False,
        )

        seen_xyz = torch.nn.functional.interpolate(
            seen_xyz.permute(2, 0, 1)[None],
            size=[112, 112],
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)

        samples = [
            [seen_xyz, seen_rgb],
            [torch.zeros((20000, 3)), torch.zeros((20000, 3))],
        ]

        seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = MCC_engine.prepare_data(
            samples, self.device, is_train=False, args=args, is_viz=True
        )
        pred_occupy = []
        pred_colors = []

        max_n_unseen_fwd = 2000

        self.model.cached_enc_feat = None
        num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_unseen_fwd))
        for p_idx in tqdm(range(num_passes)):
            p_start = p_idx     * max_n_unseen_fwd
            p_end = (p_idx + 1) * max_n_unseen_fwd
            cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
            cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
            cur_labels = labels[:, p_start:p_end].zero_()

            with torch.no_grad():
                _, pred = self.model(
                    seen_images=seen_images,
                    seen_xyz=seen_xyz,
                    unseen_xyz=cur_unseen_xyz,
                    unseen_rgb=cur_unseen_rgb,
                    unseen_occupy=cur_labels,
                    cache_enc=True,
                    valid_seen_xyz=valid_seen_xyz,
                )
            pred_occupy.append(pred[..., 0].cpu())
            if args.regress_color:
                pred_colors.append(pred[..., 1:].reshape((-1, 3)))
            else:
                pred_colors.append(
                    (
                        torch.nn.Softmax(dim=2)(
                            pred[..., 1:].reshape((-1, 3, 256)) / args.temperature
                        ) * torch.linspace(0, 1, 256, device=pred.device)
                    ).sum(axis=2)
                )

        pred_occupy = torch.cat(pred_occupy, dim=1)
        pred_colors = torch.cat(pred_colors, dim=0)
        pred_occ = torch.nn.Sigmoid()(pred_occupy).cpu()
        
        for t in score_thresholds:
            print(t)
            pos = pred_occ > t
            

            points = unseen_xyz[pos].reshape((-1, 3))
            features = pred_colors[None][pos].reshape((-1, 3))
            good_points = points[:, 0] != -100

            if good_points.sum() == 0:
                continue

            return Pointclouds(
                points=points[good_points][None].cpu(),
                features=features[good_points][None].cpu(),
            )
       
