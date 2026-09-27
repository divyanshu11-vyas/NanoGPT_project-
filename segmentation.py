import spacy
import re
import json


cleaned_text = True

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
segments = run_segmentation(cleaned_text)
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