import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

import torch
from mamba_model import MambaModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

model = MambaModel.from_pretrained(pretrained_model_name="Zyphra/BlackMamba-2.8B")
model = model.cuda().half()

def generate(prompt):
    answer = ''
    inputs = tokenizer.encode(prompt, return_tensors='pt').cuda()
    while answer.count('\n') < 3:
        out_logits = model(inputs)
        last_logits = out_logits[:, -1]
        pred_vocab = last_logits.argmax(dim=-1)
        out_token = tokenizer.decode(pred_vocab)
        print(out_token.replace('\n', '<NL>'), end='')
        answer += out_token
        inputs = torch.concat((inputs, pred_vocab.unsqueeze(0)), -1)
    return answer

while True:
    prompt = input('Input: ')
    generate(prompt)
    print()
