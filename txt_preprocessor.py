import ftfy
import chardet
from bs4 import BeautifulSoup
import re
import unicodedata
import json
import spacy
from transformers import AutoTokenizer
import hashlib
import tiktoken
from langdetect import detect, LangDetectException


def fix_encoding(path :str) -> str:

    # detecting the encoding if unsure
    with open (path ,"rb") as f:
        raw = f.read()

    detected = chardet.detect(raw)
    # chardet examines the byte sequence 
    # aakes an educated guess about its encoding or what type of encoding needs to be done
    encoding = detected["encoding"] or "utf-8"
    # this asks python to use the detectd encoding or use utf 8 by default

    text = raw.decode(encoding, errors="replace")
    # converting the encoded data into python string and replacing the error
    # characters with unicode replace character
    text = ftfy.fix_text(text)
    # fixing the mojibake which caused by using the wromg encoding

    return text




def strip_html( text: str) -> str:

    soup = BeautifulSoup(text, "html.parser")
    # detecting the html tags by selecting the html-type parser
    text = soup.get_text(separator =" ")
    # removing the html tags and put space between the substrings

    #now remove Markdown formatting that might remain after HTML extraction.
    text = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', text)  # bold/italic
    text = re.sub(r'#{1,6}\s*', '', text)               # headers
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text) # links

    return text



def normalize_unicode( text : str) -> str :

    text = unicodedata.normalize("NFKC", text) # nkfc is a form of normalization
    # converting the unicode text into standardized Unicode form

    # replacing the lame ass fancy quotes and dashes with their ascii equivalent
    replacements = {

        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-", "\u2014": "-",   # en/em dash
        "\u2026": "...",                # ellipsis
        "\u00a0": " ",                  # non-breaking space
    }
    for char , replacement in replacements.items():
        text = text.replace(char, replacement)
    return text



def remove_noise(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)  # url
    text = re.sub(r'\S+@\S+\.\S+', '', text)  # emails
    text = re.sub(r'\+?[\d\s\-\(\)]{7,15}', '', text) # phone numbers
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # non-ASCII
    text = re.sub(r'_{2,}|-{2,}|={2,}|\*{2,}', '', text) # horizontal rules
    return text
# we generally dont need non ascii if the text is multilingual


# if we have the case sesitive downstream task
def lowercase(text : str ,lowercase : bool = True) -> str:

    return text.lower() if lowercase else text 



def whitespace(text : str) -> str:

    text = re.sub(r'[ \t]+', ' ', text)        # multiple spaces/tabs → single space
    text = re.sub(r'\n{3,}', '\n\n', text)     # 3+ newlines → paragraph break
    text = re.sub(r' \n|\n ', '\n', text)      # trailing/leading space on lines
    text = text.strip()
    return text






def clean_text_file(path: str, use_lowercase: bool = False) -> str:
    text = fix_encoding(path)
    text = strip_html(text)
    text = normalize_unicode(text)
    text = remove_noise(text)
    text = lowercase(text, use_lowercase)
    text = whitespace(text)
    return text

cleaned = clean_text_file("fineweb_sample.txt", use_lowercase=False)

with open("cleaned_output.txt", "w", encoding="utf-8") as f:
    f.write(cleaned)













# SECMENTATION..............



nlp = spacy.load("en_core_web_sm")
# we should use en_core_web_sm for speed, en_core_web_trf for accuracy on complex text.
nlp.max_length = 2_000_000



def segment_sentences(text: str) -> list[str]:

    doc = nlp(text)
    # stripping hte sentences
    sentences = [sent.text.strip() for sent in doc.sents]
    # now erasing the MT strings
    sentences = [s for s in sentences if s]

    return sentences


# sents is the list that is returned by segment sentences containing the sub sentences
def segment_paragraphs(text: str) -> list[str]:

    paragraphs = re.split(r'\n\s*\n', text) # r means raw string tells Python to treat the backslashes in the string as intended for the regular expression.
    # for \n newline character
    # splits the string by the occurence of the pattern
    # \s* means zero or more whitespace characters.
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


# Take one large piece of text and turn it into a clean list where each element represents one paragraph.
def segment_all(text: str) -> list[dict]:

    paragraphs = segment_paragraphs(text)
    result = []
    for i, para in enumerate(paragraphs):

        sents = segment_sentences(para)
        result.append({
            "para_id": i,
            "paragraph": para,
            "sentences": sents,
            "sent_count": len(sents)
        })

    return result


# filtering segment removes that paragraphs which are too short, empyt and contain garbage ass symbols
def filter_segments(segments: list[dict],min_sentences: int = 1,min_chars: int = 20) -> list[dict]:

    filtered = [] # the khaali list jisme there will be filtered paragraphs
    for seg in segments:

        if len(seg["paragraph"]) < min_chars:
            continue
        if seg["sent_count"] < min_sentences:
            continue
        # calculate the alphabetic-character ratio to know what percentage of the paragraph's characters are letters
        alpha_ratio = sum(
            c.isalpha() for c in seg["paragraph"]
        ) / len(seg["paragraph"])
        # c is current character in the paragraph
        if alpha_ratio < 0.5:
            continue

        filtered.append(seg)

    return filtered


# now removing very short sentences
def clean_sentences(sentences: list[str]) -> list[str]:

    cleaned = []

    for s in sentences:

        if len(s.split()) < 3:
            continue

        if not any(c.isalpha() for c in s):
            continue

        cleaned.append(s)

    return cleaned


def run_segmentation(cleaned_text: str) -> list[dict]:

    segments = segment_all(cleaned_text)
    segments = filter_segments(segments)

    for seg in segments:

        seg["sentences"] = clean_sentences(seg["sentences"])

        seg["sent_count"] = len(seg["sentences"])
    # drop paragraphs that have no valid sentences left
    segments = [s for s in segments if s["sentences"]]

    return segments
# Usage
segments = run_segmentation(cleaned)
print(f"Total paragraphs: {len(segments)}")
print(f"Total sentences: {sum(s['sent_count'] for s in segments)}")





# saving segments to json
def save_segments(segments: list[dict], output_path: str):

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            segments,
            f,
            indent=2,
            ensure_ascii=False
        )
    print(f"Saved {len(segments)} segments to {output_path}")

save_segments(segments, "segments.json")









# CHUNKING PROCESS
# step 1
# enc = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") # llma tokenizer
with open("segments.json", "r", encoding="utf-8") as f:
    segments = json.load(f)

# Load tokenizer — match this to your target LLM
# GPT-4 / GPT-3.5 → "cl100k_base"
# GPT-2           → "gpt2"
enc = tiktoken.get_encoding("cl100k_base")





def count_tokens(text :str) -> int:
    return len(enc.encode(text))


# step 2
CHUNK_SIZE    = 512    # max tokens per chunk
CHUNK_OVERLAP = 50     # overlap tokens between chunks (preserves context)
MIN_CHUNK_TOKENS = 30  # drop chunks smaller than this




def chunk_sentences( sentences : list[str], chunk_size :int, overlap : int) -> list[str]:

    chunks = []
    current_chunk = []
    current_tokens = 0

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
                wt = count_tokens(word)


                if part_tokens +wt > chunk_size:
                    part.append(" ".join(part))
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



















# quality scoring and testing 

with open ("chunks.json", "r") as f:

    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")



def is_target_language(text: str, target_lang: str = "en") -> bool:
    try:
        return detect(text) == target_lang
    except LangDetectException:
        return False

# checking the target language english
# is_target_language("Hello world this is English text") 



def symbol_ratio(text: str) -> float:
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / len(text) if text else 1.0

# Good text is usually below 0.2
symbol_ratio("Hello world!")        # 0.05
symbol_ratio("###@@!!$$$%%%^^^")    # 1.0





def digit_ratio(text: str) -> float:
    digits = sum(1 for c in text if c.isdigit())
    return digits / len(text) if text else 1.0

# Good prose is usually below 0.15
digit_ratio("In 2024 there were 42 incidents")  # 0.08
digit_ratio("192.168.1.1 404 500 302 200")      # 0.55





def short_line_ratio(text: str, min_words: int = 3) -> float:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 1.0
    short = sum(1 for l in lines if len(l.split()) < min_words)
    return short / len(lines)

# Good prose is usually below 0.3
short_line_ratio("This is a sentence.\nAnd another one here.")  # 0.0
short_line_ratio("Home\nAbout\nContact\nBlog\nShop")            # 1.0





def has_repeated_content(text: str, threshold: float = 0.2) -> bool:
    words = text.lower().split()
    if not words:
        return True
    unique_ratio = len(set(words)) / len(words)
    return unique_ratio < threshold  # True = too repetitive

has_repeated_content("buy now buy now buy now buy now")  # True
has_repeated_content("The quick brown fox jumps over")   # False





def score_chunk(chunk: dict) -> float:
    text = chunk["text"]
    score = 1.0

    # Language check — hard fail
    if not is_target_language(text):
        return 0.0

    # Penalize high symbol ratio
    sr = symbol_ratio(text)
    if sr > 0.3:
        return 0.0
    score -= sr * 0.5

    # Penalize high digit ratio
    dr = digit_ratio(text)
    if dr > 0.4:
        return 0.0
    score -= dr * 0.3

    # Penalize short line ratio
    slr = short_line_ratio(text)
    if slr > 0.5:
        return 0.0
    score -= slr * 0.2

    # Penalize repeated content
    if has_repeated_content(text):
        return 0.0

    return round(max(score, 0.0), 4)





def run_quality_scoring(chunks: list[dict], min_score: float = 0.5) -> list[dict]:
    scored = []
    for chunk in chunks:
        score = score_chunk(chunk)
        chunk["quality_score"] = score
        scored.append(chunk)

    # Filter below threshold
    passed = [c for c in scored if c["quality_score"] >= min_score]
    failed = len(scored) - len(passed)

    print(f"Total chunks   : {len(scored)}")
    print(f"Passed scoring : {len(passed)}")
    print(f"Dropped        : {failed}")

    return passed





def save_quality_chunks(chunks: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    scores = [c["quality_score"] for c in chunks]
    print(f"Saved {len(chunks)} chunks → {output_path}")
    print(f"Avg quality score : {sum(scores)/len(scores):.3f}")
    print(f"Min quality score : {min(scores):.3f}")
    print(f"Max quality score : {max(scores):.3f}")





def run_quality_pipeline(input_path: str, output_path: str, min_score: float = 0.5):
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    # Score and filter
    quality_chunks = run_quality_scoring(chunks, min_score)
    # Save
    save_quality_chunks(quality_chunks, output_path)
    return quality_chunks

quality_chunks = run_quality_pipeline("chunks.json", "quality_chunks.json")




