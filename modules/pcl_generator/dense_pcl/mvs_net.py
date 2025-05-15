 # Import lại nếu cần
import torch
import os
from torch.utils.data import DataLoader
from utils.mvs.preprocess import save_pfm
from models.depth_map_generator.mvs_net import MVSNet
from utils.mvs.dataset import find_dataset_def
from tqdm import tqdm

def run_single_scan(scan_path, outdir, model_path, num_depth=192, interval_scale=1.06):
   
    # Cấu hình thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tạo dataset từ thư mục scan_path
    MVSDataset = find_dataset_def("dtu_dataset")
    dataset = MVSDataset(
        datapath=scan_path,
        listfile=None,
        mode="test",
        nviews=5,
        interval_scale=interval_scale
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

    # Load model
    model = MVSNet(refine=False)
    model.to(device)

    # Load checkpoint
    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    model.eval()

    # Tạo thư mục kết quả
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_est"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "confidence"), exist_ok=True)

    # Dự đoán
    with torch.no_grad():
        for idx, sample in enumerate(tqdm(dataloader)):
            sample_cuda = {
                key: value.to(device)
                for key, value in sample.items()
                if isinstance(value, torch.Tensor)
            }
            outputs = model(sample_cuda)

            # Lấy depth và confidence
            depth_est = outputs["depth"].squeeze(0).cpu().numpy()
            photometric_confidence = outputs["photometric_confidence"].squeeze(0).cpu().numpy()

            # Lưu kết quả
            filename = sample["filename"][0]  # ví dụ: 'scan1/00000000'
            name = os.path.basename(filename)
            save_pfm(os.path.join(outdir, "depth_est", f"{name}_init.pfm"), depth_est)
            save_pfm(os.path.join(outdir, "confidence", f"{name}_prob.pfm"), photometric_confidence)

    print("✅ Done processing single scan.")
if __name__ == "__main__":
    run_single_scan(
        scan_path="./your_scan/",
        outdir="./outputs/",
        model_path="models/matchers/weights/model_000014.ckpt"
    )
