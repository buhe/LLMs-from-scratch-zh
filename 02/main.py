import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

import json
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
import torch
from torch.utils.data import Dataset, DataLoader

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = ChatGLMTokenizer(vocab_file='../chatglm_tokenizer/tokenizer.model')

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0
    )

    return dataloader
class GPTDatasetV1(Dataset):
    def __init__(self, data, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        # token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        token_ids = []
        for line in tqdm(data):
            text=line['completion']
            text_id=tokenizer.encode(text,add_special_tokens=False)
            text_id.append(tokenizer.special_tokens['<eos>'])
            if len(text_id)>5:
                token_ids += text_id
            # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
if __name__=="__main__":
    vocab_size=64793
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    with open('./data/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data=json.load(f)
    dataloader = create_dataloader_v1(data, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    context_length = 4
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + pos_embeddings
    print("Final Input Embeddings:\n", input_embeddings.shape)