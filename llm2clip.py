import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from llm2vec import LLM2Vec

class LLM2CLIPWrapper:
    def __init__(self, llm_model_name, clip_model_name="openai/clip-vit-large-patch14-336"):
        # 加载LLM2CLIP组件
        config = AutoConfig.from_pretrained(llm_model_name, trust_remote_code=True)
        self.llm_model = AutoModel.from_pretrained(
            llm_model_name, 
            torch_dtype=torch.bfloat16,
            config=config,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        
        # 兼容性修复
        self.llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' 
        
        # 初始化LLM2Vec编码器
        self.encoder = LLM2Vec(
            self.llm_model,
            self.tokenizer,
            pooling_mode="mean",
            max_length=512,
            doc_max_length=512
        )
        
    def encode_text(self, texts):
        return self.encoder.encode(texts, convert_to_tensor=True)