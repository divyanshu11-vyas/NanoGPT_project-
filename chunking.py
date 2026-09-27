import json
from transformers import AutoTokenizer
import hashlib

# step 1
enc = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") # llma tokenizer

def count_tokens(text :str) -> int:
    return len(enc.decode(text))


# step 2
CHUNK_SIZE    = 512    # max tokens per chunk
CHUNK_OVERLAP = 50     # overlap tokens between chunks (preserves context)
MIN_CHUNK_TOKENS = 30  # drop chunks smaller than this




def chunk_sentences( sentences : list[str], chunk_size :int, overlap : int) -> list[str]:

    chunks = []
    current_chunk = []
    current_tokens = []

    for sent in sentences:

        sent_tokens= count_tokens(sent)

        if sent_tokens > chunk_size:

            if current_chunk: # checking if the current chunk is empty

                chunks.append(" ".join(current_chunk)) # like chunk 1 =  s1+s2
                current_chunk = []
                current_tokens = 0

            # hard split of sentence 
            words = sent.split()
            part = []
            part_tokens = 0
            for word in words:
                wt = count_tokens(words)


                if part_tokens +wt > chunk_size:
                    part.append(" ".join(part_tokens))
                    part = []
                    part_tokens = 0

                part.append(word)
                part_tokens += wt

            if part:
                chunks.append(" ".join(part))
            continue # Stop processing this sentence in the current iteration and jump directly to the next sentence.
    # Without continue, the code would continue downward and try to process the giant sentence again as a normal sentence.

        if current_tokens + sent_tokens > chunk_size:

            chunks.append(" ".join(current_chunk)) # saving the current chunk
            overlap_chunk = []
            overlap_tokens = 0

            for s in reversed(current_chunk):
                st = count_tokens(s)
                if overlap_tokens + st > overlap: # if it exceeds the overlap lmit We don't add that sentence
                    break # stop looking for more overlap
                overlap_chunk.insert(0, s)
                overlap_tokens += st

            current_chunk = overlap_chunk
            current_tokens = overlap_tokens


        current_chunk.append(sent)
        current_tokens += sent_tokens

    # last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks



def chunk_all_segments(segments : list[dict]) -> list[dict]:

    all_chunks = []
    chunk_id = 0

    for seg in segments:

        sentences = seg["sentences"]
        chunks = chunk_sentences(sentences, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id":   chunk_id,
                "para_id":    seg["para_id"],
                "chunk_index": i,
                "text":       chunk_text,
                "token_count": count_tokens(chunk_text)
            })
            chunk_id += 1

        return all_chunks

raw_chunks = chunk_all_segments(segments)
print(f"Total raw chunks: {len(raw_chunks)}")





def filter_chunks(chunks: list[dict]) -> list[dict]:
    filtered = []
    for chunk in chunks:
        text = chunk["text"]
        tokens = chunk["token_count"]

        # Too short
        if tokens < MIN_CHUNK_TOKENS:
            continue

        # Too long
        if tokens > CHUNK_SIZE:
            continue

        # Mostly non-alphabetic
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.4:
            continue

        # Too many newlines (formatting garbage)
        newline_ratio = text.count("\n") / len(text)
        if newline_ratio > 0.2:
            continue

        filtered.append(chunk)

    print(f"Kept {len(filtered)} / {len(chunks)} chunks after filtering")
    return filtered

filtered_chunks = filter_chunks(raw_chunks)





def deduplicate_chunks( chunks : list[dict]) -> list[dict]:

    seen = set()
    unique = [] # containing the chunks that survives the duplication checks

    for chunk in chunks:

        # normalizing it before hashing
        normalized = " ".join(chunk["text"].lower().split())
        # trying to make superficially different versions of the same text look identical

        hash_val = hashlib.md5(normalized.encode()).hexdigest()
        # creating the fingerprint of the normalized text
        # this first convert the text into bytes and then 
        # checkng if all the normalized fingerprints has same hash value
        # hexigest converts the hash into a readable hexadecimal string.

        if hash_val not in seen:
            seen.add(hash_val)
            unique.append(chunk)

    print(f"Removed {len(chunks) - len(unique)} duplicates")
    return unique

deduped_chunks = deduplicate_chunks(filtered_chunks)




# for near duplicate detection

# pip install datasketch
# from datasketch import MinHash, MinHashLSH

# lsh = MinHashLSH(threshold=0.85, num_perm=128)

# def get_minhash(text: str) -> MinHash:
#     m = MinHash(num_perm=128)
#     for word in text.lower().split():
#         m.update(word.encode("utf8"))
#     return m



def run_chunking(segments_path: str) -> list[dict]:
    # Load
    with open(segments_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Chunk
    raw_chunks = chunk_all_segments(segments)

    # Filter
    filtered = filter_chunks(raw_chunks)

    # Deduplicate
    final_chunks = deduplicate_chunks(filtered)

    return final_chunks

chunks = run_chunking("segments.json")


# saving the entire into JSON
def save_chunks(chunks: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks → {output_path}")

    # Print stats
    tokens = [c["token_count"] for c in chunks]
    print(f"Avg tokens/chunk : {sum(tokens)/len(tokens):.0f}")
    print(f"Min tokens/chunk : {min(tokens)}")
    print(f"Max tokens/chunk : {max(tokens)}")

save_chunks(chunks, "chunks.json")



