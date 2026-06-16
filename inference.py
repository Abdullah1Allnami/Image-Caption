import os
import torch
from torchvision import transforms
from PIL import Image

from dataset import Vocabulary
from model import ShowAndTell

def greedy_search(model, image_tensor, vocab, max_length=20, device="cpu"):
    """Generate caption using Greedy Search."""
    model.eval()
    predicted_words = []
    
    with torch.no_grad():
        # 1. Feed the image features
        features = model.encoder(image_tensor.to(device)) # Shape: (1, embed_size)
        inputs = features.unsqueeze(1)                   # Shape: (1, 1, embed_size)
        
        # LSTM forward pass (first step)
        hiddens, (h, c) = model.decoder.lstm(inputs)
        logits = model.decoder.linear(hiddens.squeeze(1)) # Shape: (1, vocab_size)
        predicted_idx = logits.argmax(dim=1).item()
        
        # 2. Iterate to generate words
        for _ in range(max_length):
            # If <end> token predicted, stop
            if predicted_idx == vocab(vocab.end_val):
                break
                
            word = vocab.idx2word.get(predicted_idx, vocab.unk_val)
            predicted_words.append(word)
            
            # Embed the generated word for the next step
            inputs = model.decoder.embed(torch.tensor([[predicted_idx]]).to(device)) # Shape: (1, 1, embed_size)
            hiddens, (h, c) = model.decoder.lstm(inputs, (h, c))
            logits = model.decoder.linear(hiddens.squeeze(1))
            predicted_idx = logits.argmax(dim=1).item()
            
    return " ".join(predicted_words)


def beam_search(model, image_tensor, vocab, beam_width=3, max_length=20, device="cpu"):
    """Generate caption using Beam Search."""
    model.eval()
    
    with torch.no_grad():
        # 1. Feed the image features
        features = model.encoder(image_tensor.to(device)) # Shape: (1, embed_size)
        inputs = features.unsqueeze(1)                   # Shape: (1, 1, embed_size)
        
        hiddens, (h, c) = model.decoder.lstm(inputs)
        logits = model.decoder.linear(hiddens.squeeze(1))
        log_probs = torch.log_softmax(logits, dim=1)
        
        # Get top-k candidates for the first step
        top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
        
        # Beam list: list of tuples (cumulative_log_prob, word_indices, (h, c))
        beams = []
        for i in range(beam_width):
            beams.append((
                top_log_probs[0, i].item(),
                [top_indices[0, i].item()],
                (h, c)
            ))
            
        # 2. Iterate to generate words
        for _ in range(max_length):
            candidates = []
            
            for score, indices, (h_prev, c_prev) in beams:
                # If sequence already reached <end>, keep it as is
                if indices[-1] == vocab(vocab.end_val):
                    candidates.append((score, indices, (h_prev, c_prev)))
                    continue
                
                # Get the last generated word and pass to LSTM
                last_word = indices[-1]
                inputs = model.decoder.embed(torch.tensor([[last_word]]).to(device))
                hiddens, (h_new, c_new) = model.decoder.lstm(inputs, (h_prev, c_prev))
                logits = model.decoder.linear(hiddens.squeeze(1))
                log_probs = torch.log_softmax(logits, dim=1)
                
                # Expand
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    candidates.append((
                        score + top_log_probs[0, i].item(),
                        indices + [top_indices[0, i].item()],
                        (h_new, c_new)
                    ))
            
            # Sort candidates by cumulative score descending and keep top-k
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
            
            # Check if all top-k sequences ended with <end>
            all_ended = all(indices[-1] == vocab(vocab.end_val) for _, indices, _ in beams)
            if all_ended:
                break
                
        # 3. Select the best sequence
        best_score, best_indices, _ = beams[0]
        
        # Convert indices to words, ignoring <end> token
        caption_words = []
        for idx in best_indices:
            if idx == vocab(vocab.end_val):
                break
            caption_words.append(vocab.idx2word.get(idx, vocab.unk_val))
            
        return " ".join(caption_words)


def greedy_search_attention(model, image_tensor, vocab, max_length=20, device="cpu"):
    """Generate caption using Greedy Search for ShowAttendAndTell."""
    model.eval()
    predicted_words = []
    
    with torch.no_grad():
        # 1. Feed the image features
        encoder_out = model.encoder(image_tensor.to(device))  # Shape: (1, num_pixels, encoder_dim)
        
        # Mean pooled features to initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(model.decoder.init_h(mean_encoder_out))
        c = torch.tanh(model.decoder.init_c(mean_encoder_out))
        
        # Initialize image embedding for step 0
        current_embed = torch.tanh(model.decoder.init_img_embed(mean_encoder_out))
        
        # 2. Iterate to generate words
        for t in range(max_length):
            # Compute attention and context
            alpha = model.decoder.attention(encoder_out, h)
            context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
            
            # LSTM step
            lstm_input = torch.cat([current_embed, context], dim=1)
            h, c = model.decoder.lstm_cell(lstm_input, (h, c))
            
            # Predict
            logits = model.decoder.fc(h)
            predicted_idx = logits.argmax(dim=1).item()
            
            # If <end> token predicted, stop
            if predicted_idx == vocab(vocab.end_val):
                break
                
            word = vocab.idx2word.get(predicted_idx, vocab.unk_val)
            predicted_words.append(word)
            
            # Embed the generated word for the next step
            current_embed = model.decoder.embed(torch.tensor([[predicted_idx]]).to(device)).squeeze(1)
            
    return " ".join(predicted_words)


def beam_search_attention(model, image_tensor, vocab, beam_width=3, max_length=20, device="cpu"):
    """Generate caption using Beam Search for ShowAttendAndTell."""
    model.eval()
    
    with torch.no_grad():
        # 1. Feed the image features
        encoder_out = model.encoder(image_tensor.to(device))  # Shape: (1, num_pixels, encoder_dim)
        
        # Mean pooled features to initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = torch.tanh(model.decoder.init_h(mean_encoder_out))
        c = torch.tanh(model.decoder.init_c(mean_encoder_out))
        
        # Initialize image embedding for step 0
        current_embed = torch.tanh(model.decoder.init_img_embed(mean_encoder_out))
        
        # Compute first step
        alpha = model.decoder.attention(encoder_out, h)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        
        lstm_input = torch.cat([current_embed, context], dim=1)
        h, c = model.decoder.lstm_cell(lstm_input, (h, c))
        
        logits = model.decoder.fc(h)
        log_probs = torch.log_softmax(logits, dim=1)
        
        # Get top-k candidates for the first step
        top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
        
        # Beam list: list of tuples (cumulative_log_prob, word_indices, (h, c))
        beams = []
        for i in range(beam_width):
            beams.append((
                top_log_probs[0, i].item(),
                [top_indices[0, i].item()],
                (h, c)
            ))
            
        # 2. Iterate to generate words
        for _ in range(max_length - 1):
            candidates = []
            
            for score, indices, (h_prev, c_prev) in beams:
                # If sequence already reached <end>, keep it as is
                if indices[-1] == vocab(vocab.end_val):
                    candidates.append((score, indices, (h_prev, c_prev)))
                    continue
                
                # Get the last generated word and pass to LSTM
                last_word = indices[-1]
                current_embed = model.decoder.embed(torch.tensor([[last_word]]).to(device)).squeeze(1)
                
                # Compute attention and context
                alpha = model.decoder.attention(encoder_out, h_prev)
                context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
                
                lstm_input = torch.cat([current_embed, context], dim=1)
                h_new, c_new = model.decoder.lstm_cell(lstm_input, (h_prev, c_prev))
                
                logits = model.decoder.fc(h_new)
                log_probs = torch.log_softmax(logits, dim=1)
                
                # Expand
                top_log_probs, top_indices = log_probs.topk(beam_width, dim=1)
                for i in range(beam_width):
                    candidates.append((
                        score + top_log_probs[0, i].item(),
                        indices + [top_indices[0, i].item()],
                        (h_new, c_new)
                    ))
            
            # Sort candidates by cumulative score descending and keep top-k
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_width]
            
            # Check if all top-k sequences ended with <end>
            all_ended = all(indices[-1] == vocab(vocab.end_val) for _, indices, _ in beams)
            if all_ended:
                break
                
        # 3. Select the best sequence
        best_score, best_indices, _ = beams[0]
        
        # Convert indices to words, ignoring <end> token
        caption_words = []
        for idx in best_indices:
            if idx == vocab(vocab.end_val):
                break
            caption_words.append(vocab.idx2word.get(idx, vocab.unk_val))
            
        return " ".join(caption_words)


def load_model_and_predict(image_path, checkpoint_path, vocab_path, beam_width=3):
    """Utility function to load checkpoints and generate captions for an image."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = ShowAndTell(
        embed_size=checkpoint["embed_size"],
        hidden_size=checkpoint["hidden_size"],
        vocab_size=checkpoint["vocab_size"],
        num_layers=checkpoint["num_layers"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0) # add batch dim
    
    # Predict
    greedy_caption = greedy_search(model, img_tensor, vocab, device=device)
    beam_caption = beam_search(model, img_tensor, vocab, beam_width=beam_width, device=device)
    
    return greedy_caption, beam_caption

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Captioning Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth", help="Path to checkpoint file")
    parser.add_argument("--vocab", type=str, default="vocab.json", help="Path to vocabulary file")
    parser.add_argument("--beam", type=int, default=3, help="Beam width for beam search")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image path {args.image} does not exist.")
    elif not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path {args.checkpoint} does not exist. Please train the model first.")
    elif not os.path.exists(args.vocab):
        print(f"Error: Vocabulary file {args.vocab} does not exist.")
    else:
        greedy, beam = load_model_and_predict(args.image, args.checkpoint, args.vocab, args.beam)
        print(f"\n--- Results for {os.path.basename(args.image)} ---")
        print(f"Greedy Caption: {greedy}")
        print(f"Beam-{args.beam} Caption: {beam}")
