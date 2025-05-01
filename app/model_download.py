import os

from transformers import AutoModel, AutoTokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

# 下载千问3大模型
model_name = "Qwen/Qwen3-1.7B"  # 替换为你想下载的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 保存到本地
save_path = "../models/qwen3"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 下载embedding模型
model_name = "moka-ai/m3e-base"  # 替换为你想下载的模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, force_download=True)

# 保存到本地
save_path = "../models/embedding"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
