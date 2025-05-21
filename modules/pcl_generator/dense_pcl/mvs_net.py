 # Import lại nếu cần
import torch
import os
from torch.utils.data import DataLoader
from utils.mvs.preprocess import write_pfm
from models.depth_map_generator.mvs_net import MVSNet
from utils.mvs.dataset import find_dataset_def
from tqdm import tqdm
import torch
from collections import OrderedDict

def load_model_without_dataparallel(model, checkpoint_path, map_location='cpu'):
    """
    Load model từ checkpoint đã được lưu với torch.nn.DataParallel
    bằng cách loại bỏ tiền tố "module." khỏi state_dict.

    Args:
        model (torch.nn.Module): kiến trúc model đã được khởi tạo.
        checkpoint_path (str): đường dẫn đến file checkpoint (.pth).
        map_location (str): thiết bị để load model ('cpu' hoặc 'cuda').

    Returns:
        model (torch.nn.Module): model đã load trọng số.
    """
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=map_location)

    # Xử lý nếu checkpoint có dict ngoài
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # Xoá tiền tố "module." nếu có
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v

    # Load trọng số
    model.load_state_dict(new_state_dict)
    return model

def run_single_scan(scan_path, outdir, model_path, num_depth=192, interval_scale=1.06):

    # Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo dataset từ thư mục scan_path
    MVSDataset = find_dataset_def("dtu_dataset")
    dataset = MVSDataset(
        datapath=scan_path,
        nviews=5,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    # Load model
    model = MVSNet(refine=False)
    model.to(device)

    # Load checkpoint
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path)
    # model.load_state_dict(state_dict["model"])
    model = load_model_without_dataparallel(model, model_path, device)
    model.eval()

    # Dự đoán
    with torch.no_grad():
      for idx, sample in enumerate(tqdm(dataloader)):
          # Chuyển tất cả các tensor sang thiết bị (CPU/GPU)
          sample_cuda = {
              key: value.to(device)
              for key, value in sample.items()
              if isinstance(value, torch.Tensor)
          }
          print(sample_cuda)
          # Chạy mô hình
          outputs = model(
              sample_cuda["imgs"],
              sample_cuda["proj_matrices"],
              sample_cuda["depth_values"]
          )

          # Lấy depth và confidence
          depth_est = outputs["depth"].squeeze(0).cpu().numpy()
          photometric_confidence = outputs["photometric_confidence"].squeeze(0).cpu().numpy()

          # Lưu kết quả
          filename = sample["filename"][0]  # ví dụ: 'scan1/00000000'
          name = os.path.basename(filename)
          write_pfm(outdir / f"{name}_init.pfm", depth_est)
          write_pfm(outdir / f"{name}_prob.pfm", photometric_confidence)

    print("✅ Done processing single scan.")


