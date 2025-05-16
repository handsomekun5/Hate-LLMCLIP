import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import copy
from transformers import CLIPModel,AutoModel
from torchmetrics import Accuracy, AUROC
from llm2clip import LLM2CLIPWrapper  # 新增导入




class MultimodalClassifier(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.fusion_mode = args.fusion_mode
        self.map_dim = args.map_dim
        self.num_pre_output_layers = args.num_pre_output_layers
        # CLIP backbone
        self.clip = CLIPModel.from_pretrained(args.clip_model)
        self.image_encoder = copy.deepcopy(self.clip.vision_model)
        self.text_encoder = copy.deepcopy(self.clip.text_model)

        if args.use_llm2clip:
            self.model = AutoModel.from_pretrained(
                "microsoft/LLM2CLIP-Openai-L-14-336", 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True).to('cuda').eval()
            self.llm2clip = LLM2CLIPWrapper(args.llm_model)
            #self.text_proj = nn.Linear(4096, 1024)  # 将LLM输出投影到CLIP文本空间
            
        if args.freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        if args.freeze_image_encoder:
            for _, p in self.image_encoder.named_parameters():
                p.requires_grad_(False)

        if args.freeze_text_encoder:
            for _, p in self.text_encoder.named_parameters():
                p.requires_grad_(False)
                
        
        if self.fusion_mode == 'align'or 'cross_dimention':
            self.fusion = CrossdimAttenton()
            # Fusion module
            image_map_layers = [nn.Linear(1024, self.map_dim), nn.Dropout(p=0.1)]
            text_map_layers = [nn.Linear(1024, self.map_dim), nn.Dropout(p=0.1)]
            self.image_proj = nn.Sequential(*image_map_layers)
            self.text_proj = nn.Sequential(*text_map_layers)
            
            pre_output_input_dim = self.map_dim
            pre_output_layers = [nn.Dropout(p=0.2)]
            pre_output_layers.extend([nn.Linear(pre_output_input_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.dropout)])
            output_input_dim = self.map_dim
            for _ in range(1, self.num_pre_output_layers): # next pre-output layers
                pre_output_layers.extend([nn.Linear(self.map_dim, self.map_dim), nn.ReLU(), nn.Dropout(p=args.dropout)])
            self.pre_output = nn.Sequential(*pre_output_layers)
            self.output = nn.Linear(output_input_dim, 1)
            """
            self.classifier = nn.Sequential(
                nn.Linear(self.map_dim*3, 1024),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(256, 2)
            )
            """
        
        elif self.fusion_mode == 'attention_m':
            self.fusion = CrossModalAttention()
            self.classifier = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                nn.Linear(256, 2)
            )
        
        self.criterion = nn.BCEWithLogitsLoss(
            reduction='mean',
            pos_weight=torch.tensor([1.5]),  # 处理类别不平衡
        )
        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc =AUROC(task='binary')

    def forward(self, pixel_values, input_ids, attention_mask,labels,text_aug):
        # Extract features
        
        if self.hparams.args.use_llm2clip:
            # 使用LLM2CLIP增强文本特征
            with torch.no_grad()，torch.cuda.amp.autocast():
                image_features = model.get_image_features(pixel_values)
                text_features = self.llm2clip.encode_text(text_aug)
                text_features = self.model.get_text_features(text_features)  # [bs, 768]
            #text_features = self.text_proj(text_features)
        else:
            image_features = self.image_encoder(pixel_values=pixel_values).pooler_output
            text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        
        """
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        text_features = self.clip.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        """
        

        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        image_features = F.normalize(image_features, p=2, dim=1) # [batch_size, d]
        text_features = F.normalize(text_features, p=2, dim=1) # [batch_size, d]
        

        if self.fusion_mode == 'align':
            # Project to common space
            #fused_features = torch.cat([torch.mul(image_features, text_features), image_features, text_features], dim=1)  # [batch_size, 3*d]
            fused_features = torch.mul(image_features,text_features)
        elif self.fusion_mode == 'attention_m':
            fused_features = self.fusion(image_features, text_features)
        elif self.fusion_mode == 'cross_dimention':
            fused_features = self.fusion(image_features, text_features)
        
        features = self.pre_output(fused_features)
        logits = self.output(features)
        

        return logits
    """
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.args.lr,
            weight_decay=self.hparams.args.weight_decay
        )
    """
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.args.lr,
            weight_decay=self.hparams.args.weight_decay
        )
        """
        # 使用余弦退火调度器
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.args.max_epochs,
                eta_min=self.hparams.args.min_lr
            ),
            'interval': 'epoch',
            'frequency': 1
        }
        """
        # 或者使用ReduceLROnPlateau
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=3,
                verbose=True
            ),
            'monitor': 'val/auc',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        logits = self(**batch)
        logits = logits.squeeze(dim=1)
        loss = self.criterion(logits, batch['labels'].float())

        self.train_acc(logits, batch['labels'])
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(**batch)
        logits = logits.squeeze(dim=1)
        loss = self.criterion(logits, batch['labels'].float())

        self.val_acc(logits, batch['labels'])
        self.val_auc(logits, batch['labels'])
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/acc', self.val_acc, prog_bar=True)
        self.log('val/auc', self.val_auc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(**batch)
        logits = logits.squeeze(dim=1)

        self.test_acc(logits, batch['labels'])
        self.test_auc(logits, batch['labels'])
        self.log('test/acc', self.test_acc, prog_bar=True)
        self.log('test/auc', self.test_auc, prog_bar=True)



class CrossdimAttenton(nn.Module):
    def __init__(self):
        super().__init__()
        dim1_project_fused_layer = [nn.Linear(1024, 512),nn.ReLU(), nn.Dropout(p=0.1)]
        dim1_project_img_layer = [nn.Linear(1024, 512),nn.ReLU(), nn.Dropout(p=0.1)]
        dim1_project_text_layer = [nn.Linear(1024, 512),nn.ReLU(), nn.Dropout(p=0.1)]

        self.dim1_project_fused = nn.Sequential(*dim1_project_fused_layer)
        self.dim1_project_img = nn.Sequential(*dim1_project_img_layer)
        self.dim1_project_text = nn.Sequential(*dim1_project_text_layer)
    
    def forward(self,img_vec,text_vec):
        fused_features_dim0 = torch.mul(img_vec,text_vec)
        
        img_vec_dim1 = self.dim1_project_img(img_vec)
        text_vec_dim1 = self.dim1_project_text(text_vec)
        fused_features_dim_project = self.dim1_project_fused(fused_features_dim0)
        fused_features_dim1 = torch.mul(img_vec_dim1,text_vec_dim1)

        return torch.cat([fused_features_dim_project,fused_features_dim1],dim=1)




class CrossModalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen_key = nn.Linear(768, 256) # 512X256
        self.gen_query = nn.Linear(768, 256)
        self.soft = nn.Softmax(dim=1)
        self.project_dense = nn.Linear(1536, 768) # 512X256
        self.drop40 = nn.Dropout(p=0.4)

    def forward(self, vec1, vec2): 
        q1 = F.relu(self.gen_query(vec1))
        k1 = F.relu(self.gen_key(vec1))
        q2 = F.relu(self.gen_query(vec2))
        k2 = F.relu(self.gen_key(vec2))
        score1 = torch.reshape(torch.bmm(q1.view(-1, 1, 256), k2.view(-1, 256, 1)), (-1, 1))
        score2 = torch.reshape(torch.bmm(q2.view(-1, 1, 256), k1.view(-1, 256, 1)), (-1, 1))
        wt_score1_score2_mat = torch.cat((score1, score2), 1)
        wt_i1_i2 = self.soft(wt_score1_score2_mat.float()) #prob
        prob_1 = wt_i1_i2[:,0]
        prob_2 = wt_i1_i2[:,1]
        wtd_i1 = vec1 * prob_1[:, None]
        wtd_i2 = vec2 * prob_2[:, None]
        out_rep = F.relu(self.project_dense(torch.cat((wtd_i1,wtd_i2), 1)))
        out_rep = self.drop40(out_rep)
        return out_rep