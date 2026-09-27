from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    streaming=True
)

with open("fineweb_sample.txt", "w", encoding="utf-8") as f:

    for i, item in enumerate(dataset):

        f.write(item["text"])
        f.write("\n\n")

        if i == 999:
            break

print("Saved 1000 documents.")

