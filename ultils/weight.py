import os
from huggingface_hub import snapshot_download

## Downloading the MCC model
def download_MCC_model():
    model_repo = "johko/mcc_co3dv2_all_categories"
    local_dir = "tmp/mcc_model"

    if os.path.exists(local_dir) and os.listdir(local_dir):  # Kiểm tra thư mục tồn tại và không rỗng
        print(f"Model already exists in: {local_dir}. Skipping download.")
    else:
        snapshot_download(repo_id=model_repo, local_dir=local_dir)
        print(f"Model downloaded to: {local_dir}")

    model_path = os.path.join(local_dir, "model.pt")
    
    # Kiểm tra xem file model có tồn tại không trước khi trả về
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    return model_path