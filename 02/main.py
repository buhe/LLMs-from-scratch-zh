import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)


from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer
if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='../chatglm_tokenizer/tokenizer.model')
    print(tokenizer.encode("你好"))