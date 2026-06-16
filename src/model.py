import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.train_cnn = train_cnn
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W) - Input images
        Returns:
            features: Tensor of shape (batch_size, embed_size) - Embedded image features
        """
        with torch.set_grad_enabled(self.train_cnn):
            features = self.resnet(images)
            features = features.view(features.size(0), -1)
        
        features = self.embed(features)
        features = self.bn(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
        Args:
            features: Tensor of shape (batch_size, embed_size) - Image features
            captions: Tensor of shape (batch_size, seq_len) - Target caption token IDs
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, vocab_size) - Output vocabulary logits
        """
        embeddings = self.dropout(self.embed(captions[:, :-1]))  # shape (batch_size, seq_len-1, embed_size)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1) # shape (batch_size, seq_len, embed_size)
        hiddens, _ = self.lstm(inputs) # shape (batch_size, seq_len, hidden_size)
        outputs = self.linear(self.dropout(hiddens))  # shape (batch_size, seq_len, vocab_size)
        return outputs 


class ShowAndTell(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, train_cnn=False):
        super(ShowAndTell, self).__init__()
        self.encoder = EncoderCNN(embed_size, train_cnn=train_cnn)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers=num_layers)

    def forward(self, images, captions):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W) - Input images
            captions: Tensor of shape (batch_size, seq_len) - Target caption token IDs
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, vocab_size) - Word logits for each step
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


class EncoderCNN_Attention(nn.Module):
    def __init__(self, encoded_image_size=14, train_cnn=False):
        super(EncoderCNN_Attention, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.train_cnn = train_cnn
        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W) - Input images
        Returns:
            features: Tensor of shape (batch_size, num_pixels, 2048) - Convolutional feature maps
        """
        with torch.set_grad_enabled(self.train_cnn):
            features = self.resnet(images)  # (batch_size, 2048, H/32, W/32)
            features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
            features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
            features = features.view(features.size(0), -1, features.size(3))  # (batch_size, num_pixels, 2048)
        return features


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Args:
            encoder_out: Tensor of shape (batch_size, num_pixels, encoder_dim)
            decoder_hidden: Tensor of shape (batch_size, decoder_dim)
        Returns:
            alpha: Attention weights of shape (batch_size, num_pixels)
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(torch.tanh(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        return alpha


class DecoderRNN_Attention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048, attention_dim=256):
        super(DecoderRNN_Attention, self).__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.init_img_embed = nn.Linear(encoder_dim, embed_size)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, encoder_out, captions):
        """
        Args:
            encoder_out: Tensor of shape (batch_size, num_pixels, encoder_dim)
            captions: Tensor of shape (batch_size, seq_len)
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)
        
        # Mean pooled features to initialize hidden state and first step input
        mean_encoder_out = encoder_out.mean(dim=1)
        
        # Initialize LSTM hidden and cell states
        h = torch.tanh(self.init_h(mean_encoder_out))
        c = torch.tanh(self.init_c(mean_encoder_out))
        
        # Initialize image embedding for step 0 input
        img_embed = torch.tanh(self.init_img_embed(mean_encoder_out))
        
        # Embed captions for teacher forcing at steps > 0
        embeddings = self.embed(captions)
        
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(encoder_out.device)
        
        for t in range(seq_len):
            if t == 0:
                current_embed = img_embed
            else:
                current_embed = embeddings[:, t-1, :]
            
            # Compute attention and context vector
            alpha = self.attention(encoder_out, h)
            context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
            
            # LSTM cell step
            lstm_input = torch.cat([current_embed, context], dim=1)
            h, c = self.lstm_cell(self.dropout(lstm_input), (h, c))
            
            # Output predictions
            preds = self.fc(self.dropout(h))
            outputs[:, t, :] = preds
            
        return outputs


class ShowAttendAndTell(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048, attention_dim=256, train_cnn=False):
        super(ShowAttendAndTell, self).__init__()
        self.encoder = EncoderCNN_Attention(train_cnn=train_cnn)
        self.decoder = DecoderRNN_Attention(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            attention_dim=attention_dim
        )

    def forward(self, images, captions):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
            captions: Tensor of shape (batch_size, seq_len)
        Returns:
            outputs: Tensor of shape (batch_size, seq_len, vocab_size)
        """
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

