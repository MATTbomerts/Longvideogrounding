import h5py
import math
import numpy as np
import tqdm
import torch
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm



def extract_sentence_feat(model_name):
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained(model_name)
    # model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # model, preprocess = clip.load("ViT-B/32")
    device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    with h5py.File("data/mad/features/CLIP_language_sentence_features.h5", 'w') as f:
        write_h5(f, "train", tokenizer, model, device)
        write_h5(f, "val", tokenizer, model, device)
        write_h5(f, "test", tokenizer, model, device)
  
    

def write_h5(h5_handler, split, tokenizer,model, device):
    qids, texts = list(), list()
    with open(f"data/mad/annotations/{split}.txt") as f:
        for line in f.readlines():
            texts.append(line.strip().split(" | ")[-1])
            qids.append(line.strip().split(" | ")[0])
    
    print(f"split: {split}, text num: {len(texts)}")
    # batch_size = 10000.0
    batch_size = 100.0
    batch_num = math.ceil(len(texts) / batch_size)

    sent_feats = list()
    # for i in range(batch_num):
    for i in tqdm(range(batch_num)):
        batches = texts[int(i*batch_size):int((i+1)*batch_size)]  #拿到一个batch的文本标注
        with torch.no_grad():
            inputs = tokenizer(batches,  #sentence_feature为什么还要加padding?
                               padding="max_length", 
                               truncation=True, 
                               max_length=77, 
                               return_tensors="pt"
                            )
            output = model(input_ids=inputs.input_ids.to(device), 
                           attention_mask=inputs.attention_mask.to(device)
                        )
            sent_feats.append(output.pooler_output.cpu().numpy())  #pooler_output输出的是句子的cls表示
    
    sent_feats = np.concatenate(sent_feats, axis=0)  #将一个batch内的特征拼接起来
    for qid, feat in zip(qids, sent_feats):
        h5_handler.create_dataset(f"{qid}", data=feat)



if __name__ == "__main__":
    # extract_sentence_feat("openai/clip-vit-base-patch32")
    extract_sentence_feat("/mnt/hdd1/zhulu/models/CLIP-B32")