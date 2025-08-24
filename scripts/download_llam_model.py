from kaggle.api.kaggle_api_extended import KaggleApi
import os

# Đường dẫn bạn muốn lưu mô hình
output_dir = "/workspace/nhitny/mcs/MoCL-NAACL-2024/pretrained/llama2-7b-hf"
os.makedirs(output_dir, exist_ok=True)

# Khởi tạo và xác thực API
api = KaggleApi()
api.authenticate()

# Tải dataset và tự động giải nén vào output_dir
api.dataset_download_files(
    dataset="lizhecheng/llama2-7b-hf",
    path=output_dir,
    unzip=True
)

print(f" Mô hình đã được lưu tại: {output_dir}")

