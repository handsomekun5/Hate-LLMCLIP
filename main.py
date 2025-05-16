import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
torch.set_float16_matmul_precision('high')  # 或 'medium'
from config import get_args
from data import get_dataloaders
from model import MultimodalClassifier

def main():
    args = get_args()
    
    # Data
    train_loader, val_loader, test_loader = get_dataloaders(args)
    
    # Model
    model = MultimodalClassifier(args)
    
    # Training
    logger = WandbLogger(project=args.exp_name, save_dir=args.log_dir)
    callbacks = [
        EarlyStopping(monitor='val/auc', patience=args.patience, mode='max'),
        ModelCheckpoint(dirpath=args.log_dir,
                       filename='best_model',
                       monitor='val/auc',
                       mode='max'),
        LearningRateMonitor(logging_interval='epoch')  # 添加这一行
    ]
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',  # 使用 GPU 加速
        devices=args.gpus,
        precision='16-mixed',
        deterministic=True,
        gradient_clip_val=args.gradient_clip_val,  # 添加梯度裁剪
        gradient_clip_algorithm="norm"  # 使用norm-based裁剪
    )
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path='best')

if __name__ == '__main__':
    main()