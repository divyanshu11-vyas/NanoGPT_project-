# import json
# from collections import Counter
# from pathlib import Path

# BASE_DIR = Path(__file__).parent

# with open(BASE_DIR / "quality_chunks.json", "r", encoding="utf-8") as f:
#     chunks = json.load(f)

# # Flatten to single text
# full_text = "\n\n".join(chunk["text"] for chunk in chunks)
# # tokenizations cannot be dont on a raw json chunk file
# # it needs a full text


# print(f"Total characters : {len(full_text):,}")
# print(f"Total words       : {len(full_text.split()):,}")
# print(f"Unique characters: {len(set(full_text)):,}")

# # saving the flat text for tokenizer training
# with open("input.txt", "w", encoding="utf-8") as f:
#     f.write(full_text)


# # build the vocabulary

# vocab = sorted(set(full_text))
# vocab_size = len(vocab)


# print(f" characters {vocab}")
# print(f"base vocab size {vocab_size}")





# # Create character → integer mapping
# char_to_int = {ch: i for i, ch in enumerate(vocab)}
# int_to_char = {i: ch for i, ch in enumerate(vocab)}




# # character level encode( baseline)
# # fucking 1 token per character

# def encode_chars(text: str) -> list[int]:
#     return [char_to_int[ch] for ch in text]

# def decode_chars(ids: list[int]) -> str:
#     return ''.join(int_to_char[i] for i in ids)




# # making my own BPE tokenizer from scratch


