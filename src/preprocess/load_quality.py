import os
import pandas as pd
import numpy as np

def generate_quality_metadata(base_dir="/home/agiuser/FypProj/data/raw/DataSet2",
                             output_path="/home/agiuser/FypProj/data/metadata_quality.csv",
                             flip_label=False):
    all_data = []
    
    # 遍历 PhysioNet 2016 的子数据集 a-f
    for c in 'abcdef':
        folder_path = os.path.join(base_dir, f'training-{c}')
        ref_csv = os.path.join(folder_path, "REFERENCE.csv")
        sqi_csv = os.path.join(folder_path, "REFERENCE-SQI.csv")
        
        if not os.path.exists(ref_csv):
            print(f"跳过: {folder_path} (找不到 REFERENCE.csv)")
            continue
        
        # 1. 加载基础信息 (文件名)
        # REFERENCE.csv 格式通常为: 文件名, 标签(-1 或 1)
        df = pd.read_csv(ref_csv, header=None, names=["fname", "diag_label"], dtype=str)
        df["fname"] = df["fname"].str.strip()
        
        # 2. 核心逻辑：加载 SQI 质量评分
        if os.path.exists(sqi_csv):
            sqi_df = pd.read_csv(sqi_csv, header=None, dtype=str)
            # 取第一列为文件名，最后一列为质量得分
            # 将得分转换为数字，处理可能的空格
            sqi_scores = pd.to_numeric(sqi_df.iloc[:, -1].str.strip(), errors='coerce')
            sqi_df = pd.DataFrame({
                "fname": sqi_df.iloc[:, 0].str.strip(), 
                "score": sqi_scores
            })
            
            # 合并诊断标签和质量评分
            # 注意：这里使用 how='inner' 确保只处理有质量评分记录的文件
            df = df.merge(sqi_df, on="fname", how="inner")
            
            # 质量合格 (score != 0) -> Label 1，质量差 (score == 0) -> Label 0
            df["label"] = (df["score"] != 0).astype(int)
            # flip_label=True：反转标签，使差质量为正类(1)，用于 SQA 训练
            if flip_label:
                df["label"] = 1 - df["label"]
        else:
            # 如果某个文件夹没有 SQI 文件，在质量模型训练中只能跳过，因为没有“真值”
            print(f"警告: {folder_path} 缺少 SQI 文件，无法生成质量标签")
            continue
        
        # 3. 补充路径信息
        df["filepath"] = df["fname"].apply(lambda x: os.path.join(folder_path, x + ".wav"))
        all_data.append(df[["fname", "label", "filepath", "score"]])

    if not all_data:
        print("错误：未找到任何有效数据，请检查路径。")
        return

    # 合并全量数据
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 打印分布情况
    print("\n" + "="*30)
    print("📊 质量评估数据集统计")
    print("-" * 30)
    print(f"总样本数: {len(final_df)}")
    if flip_label:
        print(f"质量差/正类 (Label 1): {len(final_df[final_df['label'] == 1])}")
        print(f"质量合格 (Label 0):    {len(final_df[final_df['label'] == 0])}")
    else:
        print(f"质量合格 (Label 1): {len(final_df[final_df['label'] == 1])}")
        print(f"质量差 (Label 0):   {len(final_df[final_df['label'] == 0])}")
    print("="*30)
    
    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"✅ 成功生成质量元数据: {output_path}")

if __name__ == "__main__":
    generate_quality_metadata()
    generate_quality_metadata(
        output_path="/home/agiuser/FypProj/data/metadata_quality_reversed.csv",
        flip_label=True
    )