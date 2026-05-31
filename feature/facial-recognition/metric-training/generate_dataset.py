import os
import csv
import random
import itertools


DATA_ROOT = os.path.join("Data")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_data")
VAL_DIR   = os.path.join(DATA_ROOT, "val_data")
TEST_DIR  = os.path.join(DATA_ROOT, "test_data")

MAX_PAIRS_PER_PERSON = 400
def scan_identities(root_dir):

    identity_images = {}
    for identity in sorted(os.listdir(root_dir)):
        identity_dir = os.path.join(root_dir, identity)
        if not os.path.isdir(identity_dir):
            continue
        images = [
            os.path.join(identity_dir, f)
            for f in sorted(os.listdir(identity_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if images:
            identity_images[identity] = images
    return identity_images


def generate_triplets(identity_images, max_pairs_per_person=MAX_PAIRS_PER_PERSON):

    all_identities = list(identity_images.keys())
    anchors, positives, negatives = [], [], []

    for identity in all_identities:
        imgs = identity_images[identity]
        if len(imgs) < 2:
            continue

        pairs = list(itertools.combinations(imgs, 2))
        if len(pairs) > max_pairs_per_person:
            pairs = random.sample(pairs, max_pairs_per_person)

        for img1, img2 in pairs:
            anchors.append(img1)
            positives.append(img2)

            neg_id = random.choice(all_identities)
            while neg_id == identity:
                neg_id = random.choice(all_identities)
            negatives.append(random.choice(identity_images[neg_id]))

    return anchors, positives, negatives


def save_to_csv(filename, anchors, positives, negatives):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Anchor', 'Positive', 'Negative'])
        for a, p, n in zip(anchors, positives, negatives):
            writer.writerow([a, p, n])
    print(f"  -> Saved {len(anchors):,} triplets to {filename}")


if __name__ == "__main__":
    random.seed(42)
    train_ids = scan_identities(TRAIN_DIR)

    t_a, t_p, t_n = generate_triplets(train_ids)
    # shuffle
    combined = list(zip(t_a, t_p, t_n))
    random.shuffle(combined)
    t_a, t_p, t_n = zip(*combined) if combined else ([], [], [])
    save_to_csv('train_triplets.csv', t_a, t_p, t_n)

    val_ids = scan_identities(VAL_DIR)

    v_a, v_p, v_n = generate_triplets(val_ids, max_pairs_per_person=1)
    combined = list(zip(v_a, v_p, v_n))
    random.shuffle(combined)
    v_a, v_p, v_n = zip(*combined) if combined else ([], [], [])
    save_to_csv('val_triplets.csv', v_a, v_p, v_n)

    test_ids = scan_identities(TEST_DIR)

    te_a, te_p, te_n = generate_triplets(test_ids, max_pairs_per_person=1)
    combined = list(zip(te_a, te_p, te_n))
    random.shuffle(combined)
    te_a, te_p, te_n = zip(*combined) if combined else ([], [], [])
    save_to_csv('test_triplets.csv', te_a, te_p, te_n)

    print("\nDone")
