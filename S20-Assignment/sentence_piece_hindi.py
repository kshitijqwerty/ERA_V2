# from tokenizers import Tokenizer
import tokenizers
import os


class CustomTokenizer:
    def __init__(self, file_path=None):
        self.tokenizer = tokenizers.SentencePieceBPETokenizer()
        if file_path:
            self.tokenizer.load(file_path)
    
    def calculate_compression(self, text):
        encoded = self.tokenizer.encode(text)
        return len(text) / len(encoded)
    
    def train_tokenizer(self, file, save_name="tokenizer"):
        if os.path.exists(file):
            self.tokenizer.train(file)
            self.tokenizer.save(save_name+".json")
        else:
            print(f"{file} Not Found")
    
    def get_tokens(self, text):
        encoded = self.tokenizer.encode(text)
        return [(self.tokenizer.decode([k]), k) for k in encoded.ids]


        

