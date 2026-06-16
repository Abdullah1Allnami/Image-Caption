import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

from dataset import build_vocab, get_loader, Vocabulary, Flickr8kDataset
from model import ShowAndTell

def train_model():
    # 1. Hyperparameters & Paths
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    learning_rate = 1e-3
    num_epochs = 10
    batch_size = 32
    num_workers = 0
    vocab_threshold = 2
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    root_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    captions_file = os.path.join(data_dir, "Flickr8k_text", "Flickr8k.token.txt")
    
    train_split_file = os.path.join(data_dir, "Flickr8k_text", "Flickr_8k.trainImages.txt")
    val_split_file = os.path.join(data_dir, "Flickr8k_text", "Flickr_8k.devImages.txt")
    
    vocab_path = os.path.join(project_dir, "vocab.json")
    checkpoint_dir = os.path.join(project_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 2. Vocabulary preparation
    if os.path.exists(vocab_path):
        vocab = Vocabulary.load(vocab_path)
        print(f"Loaded existing vocabulary from {vocab_path} (size: {len(vocab)})")
    else:
        vocab = build_vocab(captions_file, threshold=vocab_threshold, vocab_save_path=vocab_path)
    
    vocab_size = len(vocab)
    pad_idx = vocab(vocab.pad_val)
    
    # 3. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 4. Data Loaders with Robust Fallback for Demo Mode
    print("Preparing train data loader...")
    train_loader = get_loader(root_dir, captions_file, vocab, train_transform, batch_size, shuffle=True, num_workers=num_workers, split_file=train_split_file)
    
    print("Preparing validation data loader...")
    val_loader = get_loader(root_dir, captions_file, vocab, val_transform, batch_size, shuffle=False, num_workers=num_workers, split_file=val_split_file)
    
    # Fallback check for demo mode
    if len(train_loader.dataset) < 5 or len(val_loader.dataset) < 2:
        print("\n[INFO] Split files returned insufficient data (expected in demo mode). Splitting programmatically from available images.")
        # Load all available images
        full_dataset = Flickr8kDataset(root_dir, captions_file, vocab, transform=None, split_file=None)
        total_len = len(full_dataset)
        if total_len == 0:
            raise ValueError(f"No images found in {root_dir}. Please run download_data.py first.")
            
        train_len = int(total_len * 0.8)
        val_len = total_len - train_len
        
        # Split indexes
        indices = torch.randperm(total_len).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]
        
        # Override datasets
        class SubsetDataset(torch.utils.data.Dataset):
            def __init__(self, base_dataset, indices, transform):
                self.base_dataset = base_dataset
                self.indices = indices
                self.transform = transform
            def __len__(self):
                return len(self.indices)
            def __getitem__(self, idx):
                img_id, caption = self.base_dataset.annotations[self.indices[idx]]
                img_path = os.path.join(self.base_dataset.root_dir, img_id)
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                # tokenize
                tokens = nltk.word_tokenize(caption.lower())
                caption_indices = [self.base_dataset.vocab(token) for token in tokens]
                caption_indices.append(self.base_dataset.vocab(self.base_dataset.vocab.end_val))
                return image, torch.tensor(caption_indices)

        train_set = SubsetDataset(full_dataset, train_indices, train_transform)
        val_set = SubsetDataset(full_dataset, val_indices, val_transform)
        
        from dataset import CollateFn
        collate_fn = CollateFn(pad_idx)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
        print(f"Programmatic split created: Train={len(train_set)}, Val={len(val_set)}")

    # 5. Initialize Model, Loss, and Optimizer
    model = ShowAndTell(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_layers=num_layers,
        train_cnn=False  # Keep ResNet weights frozen
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # Optimize only parameters that require gradients (decoder + linear projection in encoder)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # 6. Training Loop
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')
    
    print(f"\nStarting training on {device} for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for images, captions, lengths in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            images = images.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            # outputs shape: (batch_size, seq_len, vocab_size)
            outputs = model(images, captions)
            
            # The targets are captions themselves
            # targets shape: (batch_size, seq_len)
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for images, captions, lengths in val_loader:
                images = images.to(device)
                captions = captions.to(device)
                
                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                
                val_loss += loss.item()
                val_steps += 1
                
        avg_val_loss = val_loss / val_steps
        
        print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        
        # Save validation plot history
        with open(os.path.join(project_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=4)
            
        # Save checkpoints
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "embed_size": embed_size,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
            "num_layers": num_layers
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest.pth")
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(checkpoint_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"New best model saved with Val Loss: {avg_val_loss:.4f}")
            
    print("Training finished successfully!")
    
    # Save Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), history["train_loss"], label="Train Loss", marker='o')
    plt.plot(range(1, num_epochs + 1), history["val_loss"], label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss history")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(project_dir, "loss_plot.png"))
    plt.close()
    print("Loss plot saved as loss_plot.png")

if __name__ == "__main__":
    train_model()
