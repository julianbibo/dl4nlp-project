import json


with open("src/annotate/annotations_en_de_full.jsonl") as f:
    annotations = [json.loads(line) for line in f]


with open("data/wmt22/en_de_train_ids.txt") as f:
    train_ids = set(f.read().splitlines())

with open("data/wmt22/en_de_eval_ids.txt") as f:
    eval_ids = set(f.read().splitlines())


final_annotations = []
skipped_train_ids = 0
in_eval_ids = 0
for a in annotations:
    if a["pair_id"] in train_ids:
        skipped_train_ids += 1
        continue
    if a["pair_id"] in eval_ids:
        in_eval_ids += 1
        final_annotations.append(a)


print(f"Skipped {skipped_train_ids} annotations that were in the training set.")
print(f"Found {in_eval_ids} annotations that were in the evaluation set.")

# save final annotations to eval_annotations_en_de.jsonl
with open("src/annotate/eval_annotations_en_de.jsonl", "w") as f:
    for a in final_annotations:
        f.write(json.dumps(a) + "\n")
