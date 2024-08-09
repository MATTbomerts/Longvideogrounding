import os
import re
import json
import csv

#使用的是MAD V1 和SNAG格式不一样
splits = ["train", "val", "test"]
root_dir = "data/mad/annotations"

for split in splits: #原本一共有三个文件
    with open(os.path.join(root_dir, "MAD_{}.json".format(split)), 'r') as f:
        raw_anns = json.load(f)


    annos = list()
    for qid, ann in raw_anns.items(): #以query为单位
        vid = ann["movie"]
        duration = ann["movie_duration"]
        spos, epos = ann["ext_timestamps"]
        query = re.sub("\n", "", ann["sentence"]) #每个视频有多条query信息
        
        annos.append([str(qid), str(vid), str(duration), str(spos), str(epos), query])

    with open("data/mad/annotations/{}.txt".format(split), 'w') as f: #每一个split生成一个txt文件
        for anno in annos:
            f.writelines(" | ".join(anno) + "\n")