import os
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
import json

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.words = []
        # Define special tokens
        self.pad_val = "<pad>"
        self.end_val = "<end>"
        self.unk_val = "<unk>"
        
        self.add_word(self.pad_val)
        self.add_word(self.end_val)
        self.add_word(self.unk_val)

    def add_word(self, word):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            self.words.append(word)

    def __call__(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk_val])

    def __len__(self):
        return len(self.word2idx)

    def save(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'idx2word': {int(k): v for k, v in self.idx2word.items()}
            }, f, indent=4)

    @classmethod
    def load(cls, filepath):
        vocab = cls()
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            vocab.word2idx = data['word2idx']
            vocab.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            vocab.words = [vocab.idx2word[i] for i in range(len(vocab.idx2word))]
        return vocab


def build_vocab(captions_file, threshold=2, vocab_save_path=None):
    """Build vocabulary from captions file."""
    print("Building vocabulary...")
    counter = Counter()
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                caption = parts[1]
                tokens = nltk.word_tokenize(caption.lower())
                counter.update(tokens)

    vocab = Vocabulary()
    # Add words that meet the frequency threshold
    for word, count in counter.items():
        if count >= threshold:
            vocab.add_word(word)

    print(f"Vocabulary built. Size: {len(vocab)}")
    if vocab_save_path:
        vocab.save(vocab_save_path)
        print(f"Vocabulary saved to {vocab_save_path}")
    return vocab


class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab, transform=None, split_file=None):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform

        # 1. Load split image names if split_file is provided
        self.split_images = None
        if split_file:
            with open(split_file, 'r', encoding='utf-8') as f:
                self.split_images = set(line.strip() for line in f if line.strip())
            print(f"Loaded split containing {len(self.split_images)} images.")

        # 2. Load and parse captions
        self.annotations = []
        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_id = parts[0].split('#')[0]
                    caption = parts[1]
                    
                    # If using split, filter images not in split
                    if self.split_images is not None and img_id not in self.split_images:
                        continue
                    
                    # Check if the image exists locally (especially relevant in demo mode)
                    img_path = os.path.join(self.root_dir, img_id)
                    if os.path.exists(img_path):
                        self.annotations.append((img_id, caption))

        print(f"Dataset initialized with {len(self.annotations)} image-caption pairs.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id, caption = self.annotations[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize and numericalize caption
        tokens = nltk.word_tokenize(caption.lower())
        caption_indices = [self.vocab(token) for token in tokens]
        caption_indices.append(self.vocab(self.vocab.end_val))

        return image, torch.tensor(caption_indices)


class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, csv_file, vocab, transform=None, split="all", split_ratio=(0.9, 0.05, 0.05)):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.split = split
        
        # Load captions from results.csv
        import pandas as pd
        df = pd.read_csv(csv_file, sep='|', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        
        # Drop any row with missing comment or image_name
        df = df.dropna(subset=['image_name', 'comment'])
        
        # Gather all unique image names
        unique_imgs = sorted(df['image_name'].unique())
        
        # Programmatic splits for reproducibility
        import random
        random.seed(42)
        random.shuffle(unique_imgs)
        
        n_imgs = len(unique_imgs)
        train_end = int(n_imgs * split_ratio[0])
        val_end = int(n_imgs * (split_ratio[0] + split_ratio[1]))
        
        if split == "train":
            allowed_imgs = set(unique_imgs[:train_end])
        elif split == "val" or split == "dev":
            allowed_imgs = set(unique_imgs[train_end:val_end])
        elif split == "test":
            allowed_imgs = set(unique_imgs[val_end:])
        else:
            allowed_imgs = set(unique_imgs)
            
        self.annotations = []
        for idx, row in df.iterrows():
            img_name = str(row['image_name']).strip()
            if img_name in allowed_imgs:
                comment = str(row['comment']).strip()
                # Local check if the image file exists
                img_path = os.path.join(self.root_dir, img_name)
                if os.path.exists(img_path):
                    self.annotations.append((img_name, comment))
                    
        print(f"Flickr30kDataset ({split}) initialized with {len(self.annotations)} image-caption pairs.")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id, caption = self.annotations[index]
        img_path = os.path.join(self.root_dir, img_id)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Tokenize and numericalize caption
        tokens = nltk.word_tokenize(caption.lower())
        caption_indices = [self.vocab(token) for token in tokens]
        caption_indices.append(self.vocab(self.vocab.end_val))

        return image, torch.tensor(caption_indices)





class CollateFn:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Sort batch by caption length (descending) - helpful for pack_padded_sequence if used
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images, dim=0)

        # Pad captions to the maximum sequence length in the batch
        lengths = [len(cap) for cap in captions]
        padded_captions = torch.zeros(len(captions), max(lengths)).long().fill_(self.pad_idx)
        for i, cap in enumerate(captions):
            padded_captions[i, :lengths[i]] = cap

        return images, padded_captions, torch.tensor(lengths)


def get_loader(root_dir, captions_file, vocab, transform, batch_size, shuffle=True, num_workers=0, split_file=None):
    dataset = Flickr8kDataset(root_dir, captions_file, vocab, transform, split_file)
    pad_idx = vocab(vocab.pad_val)
    collate_fn = CollateFn(pad_idx)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader


def get_loader_30k(root_dir, csv_file, vocab, transform, batch_size, shuffle=True, num_workers=0, split="all"):
    dataset = Flickr30kDataset(root_dir, csv_file, vocab, transform, split)
    pad_idx = vocab(vocab.pad_val)
    collate_fn = CollateFn(pad_idx)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader

