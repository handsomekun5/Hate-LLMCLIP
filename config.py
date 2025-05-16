import argparse
import torch
def get_args():
    parser = argparse.ArgumentParser(description='Multimodal Hateful Meme Classification')
    
    # Data parameters
    parser.add_argument('--data_dir', default='/root/autodl-tmp/hateful_memes', type=str)
    parser.add_argument('--image_size', type=int, default=336)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    # 可添加文本处理参数
    parser.add_argument('--use_caption', type=bool, default=True,
                    help='Whether to use caption concatenation')
    parser.add_argument('--caption_sep_token', type=str, default="[SEP]",
                    help='Separator token for text and caption')
    
    # Model parameters
    parser.add_argument('--clip_model', default='openai/clip-vit-large-patch14-336', type=str)
    parser.add_argument('--map_dim', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--freeze_clip', type=bool, default=True)
    parser.add_argument('--freeze_image_encoder', type=bool, default=True)
    parser.add_argument('--freeze_text_encoder', type=bool, default=True)
    parser.add_argument('--fusion_mode',type=str,default='align',choices=['align','attention_m','cross_dimention'])
    parser.add_argument('--num_pre_output_layers',type=int,default=1)
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=25)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--gradient_clip_val', type=float, default=1, 
                      help='Max norm of the gradients')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                    help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--use_llm2clip', type=bool, default=True,
                    help='Enable LLM2CLIP enhancement')
    parser.add_argument('--llm_model', type=str, default="microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned",
                    help='LLM2CLIP model name')
    parser.add_argument('--llm_pooling', type=str, default="mean",choices=['mean', 'max', 'cls'],
                    help='Pooling mode for LLM text features')
        
    # Experiment tracking
    parser.add_argument('--log_dir', default='/root/autodl-tmp/result_hate/clip/e1', type=str)
    parser.add_argument('--exp_name', default='hateful_memes', type=str)
    
    return parser.parse_args()