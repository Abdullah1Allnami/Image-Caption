import os
import json
import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
from torchvision import transforms
from dataset import Vocabulary
from model import ShowAndTell, ShowAttendAndTell
from inference import greedy_search, beam_search, greedy_search_attention, beam_search_attention

app = Flask(__name__)

# Check device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "best.pth")
if not os.path.exists(CHECKPOINT_PATH):
    CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoints", "latest.pth")
VOCAB_PATH = os.path.join(PROJECT_DIR, "vocab.json")

# Global variables for model and vocab
model = None
vocab = None

def get_model():
    global model, vocab
    if model is None:
        if not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(VOCAB_PATH):
            return None, None
        
        vocab = Vocabulary.load(VOCAB_PATH)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        
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
        
    return model, vocab

# Image preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/caption', methods=['POST'])
def get_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    try:
        # Load and transform image
        img = Image.open(file.stream).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get model
        net, voc = get_model()
        if net is None:
            return jsonify({'error': 'Model or Vocabulary files not found. Please train the model first.'}), 500
            
        # Get search options
        beam_w = int(request.form.get('beam_width', 3))
        max_len = int(request.form.get('max_length', 20))
        
        # Inference
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        if "attention_dim" in checkpoint:
            greedy_cap = greedy_search_attention(net, img_tensor, voc, max_length=max_len, device=device)
            beam_cap = beam_search_attention(net, img_tensor, voc, beam_width=beam_w, max_length=max_len, device=device)
        else:
            greedy_cap = greedy_search(net, img_tensor, voc, max_length=max_len, device=device)
            beam_cap = beam_search(net, img_tensor, voc, beam_width=beam_w, max_length=max_len, device=device)
        
        return jsonify({
            'success': True,
            'greedy': greedy_cap,
            'beam': beam_cap
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo-images')
def get_demo_images():
    demo_dir = os.path.join(PROJECT_DIR, "data", "Flicker8k_Dataset")
    if os.path.exists(demo_dir):
        try:
            files = sorted([f for f in os.listdir(demo_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:12]
            return jsonify({
                'success': True,
                'images': files
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True, 'images': []})

@app.route('/api/demo-image/<filename>')
def get_demo_image_file(filename):
    demo_dir = os.path.join(PROJECT_DIR, "data", "Flicker8k_Dataset")
    return send_from_directory(demo_dir, filename)

@app.route('/api/history')
def get_history():
    history_path = os.path.join(PROJECT_DIR, "history.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            return jsonify({
                'success': True,
                'history': history
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'No training history found.'}), 404

@app.route('/api/vocab')
def get_vocab():
    net, voc = get_model()
    if voc is None:
        return jsonify({'success': False, 'error': 'Vocabulary or model files not found.'}), 500
    return jsonify({
        'success': True,
        'vocab_size': len(voc),
        'words': voc.words,
        'special_tokens': {
            'pad': voc.pad_val,
            'end': voc.end_val,
            'unk': voc.unk_val
        }
    })

if __name__ == '__main__':
    print("Starting flask server on http://127.0.0.1:5003")
    app.run(host='127.0.0.1', port=5004, debug=True)
