import pandas as pd
import os

folder = "/workspace/nhitny/mcs/MoCL-NAACL-2024/datasets/wos/WOS11967"  # Đường dẫn thư mục bạn đang để file .txt

with open(os.path.join(folder, "X.txt"), encoding="utf-8") as f:
    input_data = [line.strip() for line in f]

with open(os.path.join(folder, "Y.txt"), encoding="utf-8") as f:
    label = [int(line.strip()) for line in f]

with open(os.path.join(folder, "YL1.txt"), encoding="utf-8") as f:
    label_level_1 = [int(line.strip()) for line in f]

with open(os.path.join(folder, "YL2.txt"), encoding="utf-8") as f:
    label_level_2 = [int(line.strip()) for line in f]

# Đảm bảo mọi danh sách đều có cùng độ dài
assert (
    len(input_data) == len(label) == len(label_level_1) == len(label_level_2)
), "Số dòng không khớp!"

df = pd.DataFrame(
    {
        "input_data": input_data,
        "label": label,
        "label_level_1": label_level_1,
        "label_level_2": label_level_2,
    }
)

# Ghi ra CSV không header để MoCL đọc được
output_path = os.path.join(folder, "wos.csv")
df.to_csv(output_path, index=False, header=False)

print(f"Đã tạo {output_path} với {len(df)} dòng.")
