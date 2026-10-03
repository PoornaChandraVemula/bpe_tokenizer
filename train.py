import os
import time

from bpe import BasicTokenizer

text = open("./dataset/TinyStories-valid.txt", "r", encoding="utf-8").read()
name = "basic_tokenizer_tiny_valid"

os.makedirs("models", exist_ok=True)

t0 = time.time()

tokenizer = BasicTokenizer()
tokenizer.train(text, 512, debug=True)

prefix = os.path.join("models", name)
tokenizer.save(prefix)

t1 = time.time()


print(f"Training time - {t1-t0:.2f} sec")
