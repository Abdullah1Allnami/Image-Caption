import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import corpus_bleu

from src.dataset import Vocabulary, Flickr8kDataset
from src.model import ShowAndTell, ShowAttendAndTell
from src.inference import greedy_search, beam_search, greedy_search_attention, beam_search_attention

def evaluate_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    root_dir = os.path.join(data_dir, "Flicker8k_Dataset")
    captions_file = os.path.join(data_dir, "Flickr8k_text", "Flickr8k.token.txt")
    test_split_file = os.path.join(data_dir, "Flickr8k_text", "Flickr_8k.testImages.txt")
    
    vocab_path = os.path.join(project_dir, "vocab.json")
    checkpoint_path = os.path.join(project_dir, "checkpoints", "best.pth")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Trying latest.pth...")
        checkpoint_path = os.path.join(project_dir, "checkpoints", "latest.pth")
        if not os.path.exists(checkpoint_path):
            print("Error: No checkpoints found. Please train the model first.")
            return

    # 1. Load vocab and model
    vocab = Vocabulary.load(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "attention_dim" in checkpoint:
        model = ShowAttendAndTell(
            embed_size=checkpoint["embed_size"],
            hidden_size=checkpoint["hidden_size"],
            vocab_size=checkpoint["vocab_size"],
            attention_dim=checkpoint["attention_dim"]
        ).to(device)
    else:
        model = ShowAndTell(
            embed_size=checkpoint["embed_size"],
            hidden_size=checkpoint["hidden_size"],
            vocab_size=checkpoint["vocab_size"],
            num_layers=checkpoint["num_layers"]
        ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model checkpoint from {checkpoint_path} (trained for {checkpoint['epoch']} epochs).")

    # 2. Setup transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 3. Load all captions (to match test images with all 5 ground truth captions)
    all_captions = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                img_id = parts[0].split('#')[0]
                caption = parts[1]
                if img_id not in all_captions:
                    all_captions[img_id] = []
                all_captions[img_id].append(caption)

    # 4. Determine test images (with fallback for demo mode)
    test_images = []
    if os.path.exists(test_split_file):
        with open(test_split_file, 'r', encoding='utf-8') as f:
            for line in f:
                img_id = line.strip()
                if img_id and os.path.exists(os.path.join(root_dir, img_id)):
                    test_images.append(img_id)

    # Fallback if split file resulted in empty list (demo mode)
    if len(test_images) == 0:
        print("Test split file empty or unused. Scanning image directory for evaluation...")
        all_imgs = [f for f in os.listdir(root_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Use a subset of 20 images for testing in demo mode
        test_images = sorted(all_imgs)[:30]
        
    print(f"Running evaluation on {len(test_images)} test images...")

    references = []
    hypotheses_greedy = []
    hypotheses_beam = []

    # 5. Generate captions and gather references
    for img_id in tqdm(test_images, desc="Evaluating"):
        img_path = os.path.join(root_dir, img_id)
        
        # Load and transform image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate captions
        if "attention_dim" in checkpoint:
            greedy_cap = greedy_search_attention(model, img_tensor, vocab, device=device)
            beam_cap = beam_search_attention(model, img_tensor, vocab, beam_width=3, device=device)
        else:
            greedy_cap = greedy_search(model, img_tensor, vocab, device=device)
            beam_cap = beam_search(model, img_tensor, vocab, beam_width=3, device=device)
        
        # Tokenize predictions
        hyp_greedy_tokens = nltk.word_tokenize(greedy_cap.lower())
        hyp_beam_tokens = nltk.word_tokenize(beam_cap.lower())
        
        hypotheses_greedy.append(hyp_greedy_tokens)
        hypotheses_beam.append(hyp_beam_tokens)
        
        # Gather references (the 5 ground truth captions)
        img_refs = all_captions.get(img_id, [])
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in img_refs]
        references.append(ref_tokens_list)

    # 6. Calculate BLEU Scores
    print("\n--- Evaluation Results ---")
    
    print("\n[Greedy Decoding]")
    b1_g = corpus_bleu(references, hypotheses_greedy, weights=(1.0, 0, 0, 0))
    b2_g = corpus_bleu(references, hypotheses_greedy, weights=(0.5, 0.5, 0, 0))
    b3_g = corpus_bleu(references, hypotheses_greedy, weights=(0.33, 0.33, 0.33, 0))
    b4_g = corpus_bleu(references, hypotheses_greedy, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-1: {b1_g * 100:.2f}")
    print(f"BLEU-2: {b2_g * 100:.2f}")
    print(f"BLEU-3: {b3_g * 100:.2f}")
    print(f"BLEU-4: {b4_g * 100:.2f}")

    print("\n[Beam Search Decoding (Width=3)]")
    b1_b = corpus_bleu(references, hypotheses_beam, weights=(1.0, 0, 0, 0))
    b2_b = corpus_bleu(references, hypotheses_beam, weights=(0.5, 0.5, 0, 0))
    b3_b = corpus_bleu(references, hypotheses_beam, weights=(0.33, 0.33, 0.33, 0))
    b4_b = corpus_bleu(references, hypotheses_beam, weights=(0.25, 0.25, 0.25, 0.25))
    print(f"BLEU-1: {b1_b * 100:.2f}")
    print(f"BLEU-2: {b2_b * 100:.2f}")
    print(f"BLEU-3: {b3_b * 100:.2f}")
    print(f"BLEU-4: {b4_b * 100:.2f}")

if __name__ == "__main__":
    evaluate_model()
