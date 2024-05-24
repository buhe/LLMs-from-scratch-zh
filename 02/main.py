import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import json
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer

if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='../chatglm_tokenizer/tokenizer.model')
    # print(tokenizer.encode("你好"))
    with open('./data/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data=json.load(f)
    # txt = data[0]['completion']
    # print(tokenizer.encode(txt))
    # print(tokenizer.encode(txt, add_special_tokens=False))
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    print(len(doc_ids))