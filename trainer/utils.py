import gc
import torch
import torchaudio
from pypesq import pesq


def load_sample(file):
    waveform, _ = torchaudio.load(file)
    return waveform


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def wsdr_fn(x_, y_pred_, y_true_, n_fft, hop_length, eps=1e-8):
    """
    Loss function
        x_      :   noisy speech
        y_pred_ :   pred speech
        y_true_ :   clean speech
    """

    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    x_ = torch.squeeze(x_, 1)
    y_true = torch.istft(y_true_, n_fft=n_fft, hop_length=hop_length, normalized=True)
    x = torch.istft(x_, n_fft=n_fft, hop_length=hop_length, normalized=True)

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true ** 2, dim=1) / (torch.sum(y_true ** 2, dim=1) + torch.sum(z_true ** 2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)


def pesq_metric(y_hat, bd, sample_rate):
    # PESQ
    with torch.no_grad():
        y_hat = y_hat.cpu().numpy()
        y = bd['y'].cpu().numpy()  # target signal

        sum = 0
        for i in range(len(y)):
            sum += pesq(y[i, 0], y_hat[i, 0], sample_rate)

        sum /= len(y)
        return torch.tensor(sum)