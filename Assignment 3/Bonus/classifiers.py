import torch
import torch.nn as nn

def build_conv(window, in_features, out_features, seq_length):
    return nn.Sequential(
        nn.Conv1d(in_features, out_features, kernel_size=window),
        nn.ReLU(),
        nn.MaxPool1d(seq_length - window + 1)
    )

class ConvnetClassifier(nn.Module):
    def __init__(self, n_words, seq_length):
        super(ConvnetClassifier, self).__init__()
        self.features3 = build_conv(3, n_words, 100, seq_length)
        self.features4 = build_conv(4, n_words, 100, seq_length)
        self.features5 = build_conv(5, n_words, 100, seq_length)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(300, 2)
        )

    def forward(self, x):
        f3 = self.features3(x)
        f4 = self.features3(x)
        f5 = self.features3(x)
        f = torch.cat((f3, f4, f5), dim=1)
        return self.classifier(f.view((-1, 1, 300))).view((-1, 2))