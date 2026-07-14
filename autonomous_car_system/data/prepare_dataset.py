import os
import json
import cv2
import numpy as np
import shutil
import random
from glob import glob

SOURCE_DIR = '../dataset'
OUTPUT_DIR = '../my_dataset'
LABELS_FILE = '../labels.txt'
VAL_SPLIT = 0.2
SEED = 42


def read_labels(label_file):
    label_to_index = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            label = line.strip()
            if label:
                label_to_index[label] = idx
    return label_to_index


def generate_mask(json_path, label_to_index):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    h = data['imageHeight']
    w = data['imageWidth']

    mask = np.zeros((h, w), dtype=np.uint8)

    shapes = data['shapes']
    shapes.sort(key=lambda x: label_to_index.get(x['label'], 0))

    for shape in shapes:
        label = shape['label']
        if label not in label_to_index:
            print(f"  Warning: label '{label}' not in {LABELS_FILE}. Skipping.")
            continue
        index = label_to_index[label]
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=index)

    img_filename = data['imagePath']
    img_path = os.path.join(os.path.dirname(json_path), img_filename)
    return img_path, mask


def main():
    random.seed(SEED)

    if not os.path.exists(LABELS_FILE):
        print(f"Error: '{LABELS_FILE}' not found")
        return

    label_to_index = read_labels(LABELS_FILE)
    print(f"Loaded labels: {label_to_index}")

    json_files = glob(os.path.join(SOURCE_DIR, '**', '*.json'), recursive=True)
    if not json_files:
        print(f"No JSON files found in '{SOURCE_DIR}'")
        return

    print(f"Found {len(json_files)} JSON files. Processing...")

    dataset_items = []
    for j_path in json_files:
        try:
            img_path, mask = generate_mask(j_path, label_to_index)
            if not os.path.exists(img_path):
                img_path = j_path.rsplit('.', 1)[0] + '.jpg'
                if not os.path.exists(img_path):
                    img_path = j_path.rsplit('.', 1)[0] + '.png'
            if not os.path.exists(img_path):
                print(f"  Warning: No source image for '{j_path}'")
                continue
            dataset_items.append({'img_path': img_path, 'mask': mask})
        except Exception as e:
            print(f"  Error processing '{j_path}': {e}")

    print(f"Successfully parsed {len(dataset_items)} items.")

    random.shuffle(dataset_items)
    val_size = max(1, int(len(dataset_items) * VAL_SPLIT))
    val_items = dataset_items[:val_size]
    train_items = dataset_items[val_size:]

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'))
        os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'))

    def save_split(items, split_name):
        print(f"Saving {len(items)} images to {split_name}...")
        for idx, item in enumerate(items, 1):
            ext = os.path.splitext(item['img_path'])[1]
            new_img_name = f"{idx}{ext}"
            new_mask_name = f"{idx}.png"
            out_img = os.path.join(OUTPUT_DIR, split_name, 'images', new_img_name)
            out_mask = os.path.join(OUTPUT_DIR, split_name, 'labels', new_mask_name)
            shutil.copy2(item['img_path'], out_img)
            cv2.imwrite(out_mask, item['mask'])

    save_split(train_items, 'train')
    save_split(val_items, 'val')

    print(f"Done! Dataset ready at '{OUTPUT_DIR}'")


if __name__ == '__main__':
    main()
