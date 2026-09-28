import tiktoken
import json
import numpy as np
import re
import unicodedata




with open("quality_chunks.json","r", encoding="utf-8") as f:

    chunks = json.load(f)


print(f"total chunks loaded = {len(chunks)}")

EOT = "<|endoftext|>"  # GPT-4 document separator
# separator between training texts
# creating a boundary marker between independent pieces of text.


# Join every chunk with EOT token
full_text = EOT.join(chunk["text"] for chunk in chunks)

# Add EOT after every chunk
full_text = full_text + EOT

print(f"Total characters: {len(full_text):,}")
# just for sanity check lol


# a few final low-level text-normalization operations before sending text to a GPT-4-style tokenizer
def clean_for_gpt4(text: str) -> str:
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Windows       → \r\n
    # Linux/macOS   → \n
    # old Mac       → \r


    # Remove null bytes — breaks tokenizer
    # Null characters can appear because of broken extraction, binary contamination, encoding/conversion issues, etc
    text = text.replace("\x00", "")


    # Remove other control characters except newline and tab
    text = re.sub(r'[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Normalize unicode to NFC (cl100k_base expects this)
    # NFC normalization tries to put Unicode into a consistent composed representation where possible
    text = unicodedata.normalize("NFC", text)

    # Collapse excessive newlines
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    return text

full_text = clean_for_gpt4(full_text)
print(f"Cleaned text length: {len(full_text):,}")






# encoding 
enc  = tiktoken.get_encoding("cl100k_base") # loadinf the tokenizer

print(f"Vocab size      : {enc.n_vocab:,}") # tells us how many token IDs exists in the text
print(f"EOT token ID    : {enc.eot_token}") # EOT becomes one token ID, not 13 separate character tokens.

print("Encoding full dataset... (may take a while for large text)")

token_ids = enc.encode( full_text, allowed_special={"<|endoftext|>"})

print(f"Total tokens    : {len(token_ids):,}")
print(f"Tokens/char     : {len(token_ids)/len(full_text):.3f}")




# verify the token distribution
# not important 

# def verify_token_distribution(token_ids: list[int], enc):
#     total = len(token_ids)

#     # Count EOT tokens
#     eot_id = enc.eot_token
#     eot_count = token_ids.count(eot_id)
#     print(f"\n--- Token Distribution ---")
#     print(f"Total tokens        : {total:,}")
#     print(f"EOT tokens          : {eot_count:,}")
#     print(f"Avg tokens/document : {total/eot_count:.0f}")

#     # Token frequency distribution
#     freq = Counter(token_ids)
#     most_common = freq.most_common(10)
#     print(f"\nTop 10 most common tokens:")
#     for tid, count in most_common:
#         decoded = enc.decode([tid])
#         print(f"  ID {tid:6d} | '{decoded}' | count: {count:,}")

#     # Check for anomalies
#     unique_tokens = len(freq)
#     coverage = unique_tokens / enc.n_vocab * 100
#     print(f"\nUnique tokens used  : {unique_tokens:,}")
#     print(f"Vocab coverage      : {coverage:.1f}%")

#     # Warn if coverage is too low
#     if coverage < 10:
#         print(" WARNING: Very low vocab coverage — dataset may be too small")
#     elif coverage > 80:
#         print(" Good vocab coverage")

# verify_token_distribution(token_ids, enc)



def split_dataset( token_ids : list[int]) -> tuple:

    n = len( token_ids)
    #total tokens the entire dataset

    # train val test
    train_end = int(n* 0.90)
    val_end = int(n*0.95)


    train_ids = token_ids[:train_end]
    val_ids = token_ids[train_end:val_end]
    test_ids = token_ids[val_end:]

    print("\n.........dataset split.........")

    print(f"Train tokens : {len(train_ids):,}  ({len(train_ids)/n*100:.1f}%)")
    print(f"Val tokens   : {len(val_ids):,}   ({len(val_ids)/n*100:.1f}%)")
    print(f"Test tokens  : {len(test_ids):,}   ({len(test_ids)/n*100:.1f}%)")

    return train_ids, val_ids, test_ids

# the tuple contains three lists and each list for train, validation and test ids as well

train_ids, val_ids, test_ids = split_dataset(token_ids)








def save_bin_files( train_ids, val_ids,test_ids):

    splits = {

        "train.bin" : train_ids,
        "val.bin" : val_ids,
        "test.bin" : test_ids

    }

    for filename, ids in splits.items():

        arr = np.array(ids, dtype=np.uint32)
        arr.tofile(filename)


        size_mb = arr.nbytes / 1024 / 1024
        print(f"Saved {filename}: {len(ids):,} tokens ({size_mb:.1f} MB)")

save_bin_files(train_ids, val_ids, test_ids)



# verifying the sets of data which takes the encoding type
# why does the verification need tokenizer
# -> Is this token ID valid?
# -> Can these IDs be decoded back into text?
def verify_bin_files(enc): 

     splits = ["train.bin", "val.bin", "test.bin"]

     print("========verifying saved files ===========")

     for filename in splits:

        arr = np.fromfile(filename, dtype=np.uint32)
        # The reader needs to interpret the bytes the same way they were written.


        assert len(arr) > 0,            f"{filename} is empty" # This checks that the binary file actually contains tokens.
        assert arr.max() < enc.n_vocab, f"Token ID exceeds vocab size in {filename}"
        # assert arr.min() >= 0,          f"Negative token ID found in {filename}"


        # Decode first 20 tokens as sanity check
        sample = enc.decode(arr[:20].tolist())
        print(f"\n{filename}")
        print(f"  Tokens  : {len(arr):,}")
        print(f"  Max ID  : {arr.max():,}")
        print(f"  Min ID  : {arr.min():,}")
        print(f"  Sample  : '{sample[:80]}...'")


print(" all files verified successfully")

verify_bin_files(enc)

# verifies that the .bin files you just created weren't corrupted or created incorrectly






def save_metadata(enc, token_ids, train_ids, val_ids, test_ids):

    meta = {
        "vocab_size": enc.n_vocab,
        "eot_token": enc.eot_token,
        "encoding": "cl100k_base",

        "total_tokens": len(token_ids),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "test_tokens": len(test_ids),

        "dtype": "uint32"
    }

    with open("meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n--- Saved meta.json ---")

    for key, value in meta.items():
        print(f"  {key}: {value}")










def run_full_data_prep(chunks_path: str):

    print("=" * 50)
    print("LLM Data Preparation Pipeline")
    print("=" * 50)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Step 1: Loaded {len(chunks):,} chunks")


    EOT = "<|endoftext|>"

    full_text = EOT.join(
        chunk["text"] for chunk in chunks
    ) + EOT

    print(f"Step 2: Joined chunks ({len(full_text):,} characters)")


    full_text = clean_for_gpt4(full_text)

    print(f"Step 3: Cleaned text ({len(full_text):,} characters)")


    enc = tiktoken.get_encoding("cl100k_base")

    token_ids = enc.encode(
        full_text,
        allowed_special={"<|endoftext|>"}
    )

    print(f"Step 4: Encoded {len(token_ids):,} tokens")
    print(f"        Vocabulary size: {enc.n_vocab:,}")
    print(f"        EOT token ID: {enc.eot_token}")


    train_ids, val_ids, test_ids = split_dataset(token_ids)

    print("Step 5: Dataset split complete")


    save_bin_files(
        train_ids,
        val_ids,
        test_ids
    )

    print("Step 6: Binary files saved")


    verify_bin_files(enc)

    print("Step 7: Binary files verified")

    save_metadata(
        enc,
        token_ids,
        train_ids,
        val_ids,
        test_ids
    )

    print("Step 8: Metadata saved")

    print("\n" + "=" * 50)
    print("DATA PREPARATION COMPLETE")
    print("=" * 50)

    print("Output files:")
    print("  train.bin  -> 90% training data")
    print("  val.bin    -> 5% validation data")
    print("  test.bin   -> 5% test data")
    print("  meta.json  -> dataset/tokenizer information")

    print("=" * 50)


if __name__ == "__main__":
    run_full_data_prep("quality_chunks.json")


