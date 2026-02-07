# import os
# import librosa
# import pandas as pd
# import numpy as np

# # 根目录
# BASE_DIR = "/mnt/d/FypProj/data/raw/DataSet1"

# def clean_fname(fname):
#     """
#     去掉 fname 前面的 'set_a/' 或 'set_b/' 前缀
#     """
#     if fname.startswith("set_a/"):
#         return fname[len("set_a/"):]
#     if fname.startswith("set_b/"):
#         return fname[len("set_b/"):]
#     return fname

# def load_metadata():
#     # --- 1. 读取 CSV ---
#     df_a = pd.read_csv(os.path.join(BASE_DIR, "set_a.csv"))
#     df_b = pd.read_csv(os.path.join(BASE_DIR, "set_b.csv"))

#     df_a["source"] = "A"
#     df_b["source"] = "B"

#     df = pd.concat([df_a, df_b], ignore_index=True)

#     # --- 2. 清洗 fname ---
#     df["fname"] = df["fname"].apply(clean_fname)

#     # --- 3. 为每一行生成真实文件路径 ---
#     def build_path(row):
#         folder = "set_a" if row["source"] == "A" else "set_b"
#         fp = os.path.join(BASE_DIR, folder, row["fname"])
#         if os.path.exists(fp):
#             return fp
#         print(f"[WARN] File not found: {fp}")
#         return None
        
#     df["filepath"] = df.apply(build_path, axis=1)

#     # 删除找不到文件的行
#     df = df.dropna(subset=["filepath"]).reset_index(drop=True)

#     # --- 4. 计算时长 ---
#     durations = []
#     for fp in df["filepath"]:
#         try:
#             y, sr = librosa.load(fp, sr=None)  # 保留原采样率
#             durations.append(len(y) / sr)
#         except Exception as e:
#             print(f"[WARN] Failed loading {fp}: {e}")
#             durations.append(np.nan)

#     df["duration"] = durations
#     return df


# if __name__ == "__main__":
#     df = load_metadata()

#     print(df.head())
#     print("总样本数:", len(df))

#     output_path = "/mnt/d/FypProj/data/metadata1.csv"
#     df.to_csv(output_path, index=False, encoding="utf-8")

#     print(f"已保存 metadata 到: {output_path}")

import os
import pandas as pd

def load_metadata(base_dir="/mnt/d/FypProj/data/raw/Dataset2"):
    all_data = []
    
    for c in 'abcdef':
        folder_path = os.path.join(base_dir, f'training-{c}')
        ref_csv = os.path.join(folder_path, "REFERENCE.csv")
        sqi_csv = os.path.join(folder_path, "REFERENCE-SQI.csv")
        
        if not os.path.exists(ref_csv): continue
        
        # 1. 加载基础标签 (ID, Label)
        df = pd.read_csv(ref_csv, header=None, names=["fname", "label"], dtype=str)
        df["fname"] = df["fname"].str.strip()
        
        # 2. 存在 SQI 则进行质量过滤
        if os.path.exists(sqi_csv):
            sqi_df = pd.read_csv(sqi_csv, header=None, dtype=str)
            # 取最后一列作为 score 并转为数字
            sqi_scores = pd.to_numeric(sqi_df.iloc[:, -1].str.strip(), errors='coerce')
            sqi_df = pd.DataFrame({"fname": sqi_df.iloc[:, 0].str.strip(), "score": sqi_scores})
            
            # 合并并剔除评分为 0 的行
            df = df.merge(sqi_df, on="fname", how="inner")
            df = df[df["score"] != 0]
        
        # 3. 补充路径信息
        df["filepath"] = df["fname"].apply(lambda x: os.path.join(folder_path, x + ".wav"))
        all_data.append(df)

    # 合并全量数据
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 4. 标签重映射: -1(Normal)->0, 1(Abnormal)->1
    final_df["label"] = final_df["label"].str.strip().map({'-1': 0, '1': 1})
    
    return final_df.dropna(subset=["label"]).reset_index(drop=True)

if __name__ == "__main__":
    df = load_metadata()
    print(f"✅ 处理完成，有效样本数: {len(df)}")
    print(df["label"].value_counts())
    df.to_csv("/mnt/d/FypProj/data/metadata_physionet.csv", index=False)