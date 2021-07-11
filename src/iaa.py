import json
import jsonlines
from sklearn.metrics import cohen_kappa_score


def load_annotations(path):
    annotator1 = dict()
    annotator2 = dict()

    with open(path, 'r', encoding='utf8') as jsonfile:
        for sample in jsonfile:
            anno = json.loads(sample)
            if anno['_session_id'] == 'iaa2-Jona':
                if anno['answer'] == 'accept':
                    annotator1[anno['_input_hash']] = anno['accept'][0]
                else:
                    annotator1[anno['_input_hash']] = 'reject'
            else:
                if anno['answer'] == 'accept':
                    annotator2[anno['_input_hash']] = anno['accept'][0]
                else:
                    annotator2[anno['_input_hash']] = 'reject'


    print(len(annotator1), len(annotator2))
    return annotator1, annotator2


def save_output(annotator1, annotator2):
    annotations1 = []
    annotations2 = []
    other_annotations1 = []
    other_annotations2 = []
    all_mentions = 0
    all_linked_mentions = 0

    for sample in annotator2:
        if sample in annotator1:
            a1 = annotator1[sample]
            a2 = annotator2[sample]
            other_annotations1.append(a1)
            other_annotations2.append(a2)
            if a1 == a2:
                all_mentions += 1
            if a1 not in ["NIL_notanorg", "NIL_otherentity", "NIL_ambiguous", "reject"]:
                if a2 not in ["NIL_notanorg", "NIL_otherentity", "NIL_ambiguous", "reject"]:
                    annotations1.append(a1)
                    annotations2.append(a2)
                    if a1 == a2:
                        all_linked_mentions += 1

    mutually_annotated = len(set(annotator1).intersection(set(annotator2)))
    labels = 1 / len(set(other_annotations1).union(set(other_annotations2)))
    print(f"{mutually_annotated} mutually annotated data samples.")
    print(f"{all_mentions} mentions annotated the same.")
    print(f"Inter-annotator agreement: {round(all_mentions/mutually_annotated, 3)}")
    cohen1 = cohen_kappa_score(other_annotations1, other_annotations2)
    print(f"Cohen's Kappa: {round(cohen1, 3)}")

    print()
    mutually_annotated_linked = len(annotations1)
    print(f"{mutually_annotated_linked} mutually annotated samples linked to KvK-number.")
    print(f"{all_linked_mentions} linked mentions annotated the same.")
    print(f"Inter-annotator agreement: {round(all_linked_mentions / mutually_annotated_linked, 3)}")
    cohen2 = cohen_kappa_score(annotations1, annotations2)
    print(f"Cohen's Kappa: {round(cohen2, 3)}")


def compute_iaa(path):
    annotator1, annotator2 = load_annotations(path)
    save_output(annotator1, annotator2)


def add_annotations(path):
    annotations = "../data/prodigy_data/annotations+iaa_output.jsonl"
    count = 0
    add = dict()

    writer = jsonlines.open(annotations, 'a')

    with open(path, 'r', encoding='utf8') as jsonfile:
        for line in jsonfile:
            sample = json.loads(line)
            if sample['_session_id'] == 'iaa2-Jona' and sample['accept']:
                id = sample['_input_hash']
                add[id] = sample
            elif sample['accept']:
                id2 = sample['_input_hash']

                if sample['accept'] and id2 in add:
                    sample2 = sample['accept'][0]
                    sample1 = add[id2]['accept'][0]

                    if sample1 == sample2:
                        if sample1 not in ["NIL_notanorg", "NIL_otherentity", "NIL_ambiguous", "reject"]:
                            annotation = add[id2]
                            annotation["_session_id"] = 'iaa-Rogier'
                            writer.write(annotation)
                            count += 1

    writer.close()

    print(f"{count} data samples added to multi-cand+iaa.json.")


def main():
    path = "../data/prodigy_data/iaa_output.jsonl"
    compute_iaa(path)
    add_annotations(path)


if __name__ == "__main__":
    main()
