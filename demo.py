from PIL import Image
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers import CLIPImageProcessor
import torch
from llm2vec import LLM2Vec
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
model_name_or_path = "microsoft/LLM2CLIP-Openai-L-14-336" # or /path/to/local/LLM2CLIP-Openai-L-14-336
model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True).to('cuda').eval()

llm_model_name = 'microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned'
config = AutoConfig.from_pretrained(
    llm_model_name, trust_remote_code=True
)
llm_model = AutoModel.from_pretrained(llm_model_name, torch_dtype=torch.bfloat16, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct' #  Workaround for LLM2VEC
l2v = LLM2Vec(llm_model, tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

captions = ["a diagram", "a dog", "a cat"]
image_path = "CLIP.png"

image = Image.open(image_path)
input_pixels = processor(images=image, return_tensors="pt").pixel_values.to('cuda')
text_features = l2v.encode(captions, convert_to_tensor=True).to('cuda')

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.get_image_features(input_pixels)
    text_features = model.get_text_features(text_features)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)
