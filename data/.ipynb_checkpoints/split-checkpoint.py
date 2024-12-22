import pandas as pd
import os
from sklearn.model_selection import train_test_split

# CSV 파일 로드
output_dir = "./data/processed"
csv_path = os.path.join(output_dir, "results.csv")
df = pd.read_csv(csv_path)

# Bounding Box가 예측된 데이터만 필터링
bbox_df = df[df["x1"].notnull()]

# fraud와 normal 별로 나누기
fraud_df = bbox_df[bbox_df["folder"] == "fraud"]
normal_df = bbox_df[bbox_df["folder"] == "normal"]

# Train, Valid, Test 분할 비율 설정
def split_data(df, folder_name, output_dir):
    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    valid, test = train_test_split(temp, test_size=0.5, random_state=42)

    # 결과를 각각 CSV로 저장
    split_dir = os.path.join(output_dir, folder_name)
    os.makedirs(split_dir, exist_ok=True)
    train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    valid.to_csv(os.path.join(split_dir, "valid.csv"), index=False)
    test.to_csv(os.path.join(split_dir, "test.csv"), index=False)

    print(f"{folder_name}: train={len(train)}, valid={len(valid)}, test={len(test)})")

# fraud와 normal에 대해 데이터 분할 수행
split_data(fraud_df, "fraud", output_dir)
split_data(normal_df, "normal", output_dir)

print("데이터 분할 완료!")

#37 41 59 92 189