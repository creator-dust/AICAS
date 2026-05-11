import argparse
import json
import os
from pathlib import Path
from datasets import load_dataset

def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert a local ScienceQA parquet dataset into the JSONL format used by quant_eval_smolvlm.py."
    )
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=script_dir / "benchmarks" / "scienceqa" / "ScienceQA-IMG",
        help="Directory containing the local parquet dataset.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=script_dir / "real_scienceqa_eval.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--image-out-dir",
        type=Path,
        default=script_dir / "scienceqa_images",
        help="Directory used to save extracted images.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of image-containing samples to export.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load from the parquet directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=script_dir / ".hf_cache",
        help="Local Hugging Face datasets cache directory.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    parquet_dir = args.parquet_dir.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    image_out_dir = args.image_out_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    if not parquet_dir.exists():
        raise FileNotFoundError(f"Parquet dataset directory not found: {parquet_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))

    print("正在加载 Parquet 数据集 (这可能需要几秒钟)...")
    # 使用 datasets 库加载本地 parquet 文件，它会自动帮我们解码图片
    dataset = load_dataset(
        "parquet",
        data_dir=str(parquet_dir),
        split=args.split,
        cache_dir=str(cache_dir / "datasets"),
    )
    
    image_out_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    
    jsonl_data = []
    saved_count = 0
    
    print(f"开始提取数据，目标提取 {args.max_samples} 条包含图片的样本...")
    
    for i, row in enumerate(dataset):
        if saved_count >= args.max_samples:
            break
            
        # 1. 过滤：我们只评测“视觉语言模型”，所以没有图片的纯文本题直接跳过
        if row.get('image') is None:
            continue
            
        # 2. 保存图片到本地硬盘
        image = row['image']
        image_filename = f"scienceqa_{i}.jpg"
        image_path = image_out_dir / image_filename
        
        # 将 RGBA 转换为 RGB（防止保存 JPG 报错）
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(image_path)
        
        # 3. 提取题目、选项和答案
        question = row.get('question', '')
        choices = row.get('choices', [])
        # ScienceQA 的 answer 通常是个数字索引 (比如 2 代表选 C)
        answer_idx = row.get('answer', 0) 
        
        # 将数字索引转换为具体的文本答案，方便你的测试脚本进行核对
        answer_text = choices[answer_idx] if choices else str(answer_idx)
        
        # 4. 组装成你的 quant_eval_smolvlm.py 规定的格式
        sample = {
            "id": f"scienceqa_{i}",
            "image": str(image_path),  # 绝对路径
            "question": question,
            "choices": choices,
            "answer": answer_text
        }
        jsonl_data.append(sample)
        saved_count += 1
        
        if saved_count % 50 == 0:
            print(f"已提取 {saved_count} 条...")

    # 5. 写入 JSONL 文件
    with output_jsonl.open('w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n转换完成！")
    print(f"图片已保存至: {image_out_dir}")
    print(f"配置文件已生成: {output_jsonl}")
    print(f"现在你可以运行你的 fp32 基线测试了！")

if __name__ == "__main__":
    main()
