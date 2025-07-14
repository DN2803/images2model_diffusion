# Images2Model with Diffusion

Dự án này trình bày pipeline tái tạo mô hình 3D từ ảnh đầu vào thông qua các bước:
1. **Trích xuất điểm đặc trưng (keypoints)**
2. **Khớp ảnh bằng phương pháp sử dụng mạng khuếch tán và tái dựng đám mây điểm**

3. **Tái dựng lưới 3D từ đám mây điểm sử dụng [Point2Mesh](https://github.com/DN2803/Surface-Reconstruction-from-Point-Cloud-Point2Mesh.git)**

## 📂 Cấu trúc dự án

images2model_diffusion/

├── utils/
│ └── helper_functions.py
├── requirements.txt
└── README.md


## 🚀 Cài đặt

### 1. Clone dự án
```bash
git clone https://github.com/DN2803/images2model_diffusion.git
cd images2model_diffusion
```
### 2. Cài đặt môi trường Python
```bash
conda create -n i2m_diffusion python=3.8 -y
conda activate i2m_diffusion
pip install -r requirements.txt
```
### 3. Cài đặt Point2Mesh (P2M)
```bash
git clone https://github.com/DN2803/Surface-Reconstruction-from-Point-Cloud-Point2Mesh.git
```

### 4. Tích hợp với pipeline chính
Quay lại repo images2model_diffusion, bạn có thể gọi Point2Mesh từ script run_point2mesh.py với dữ liệu từ bước trước

## 📌 Yêu cầu phần cứng

GPU >= 8GB VRAM (khuyến nghị sử dụng NVIDIA CUDA)

RAM >= 16GB

## 💡 Đóng góp
Mọi đóng góp đều được chào đón! Hãy gửi pull request hoặc tạo issue nếu bạn gặp lỗi hoặc có ý tưởng cải tiến.

## 📬 Liên hệ
Tác giả: DN2803, @minhtuan13
Email: [ntkngann2k3@gmail.com]