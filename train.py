import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
from typing import Tuple

class EuroSATDataset(Dataset):
    """Dataset class for EuroSAT satellite imagery"""
    
    def __init__(self, root_dir: str, transform=None):
        from eurosat import train as eurosat_train


        def main():
            """Wrapper that calls package training entrypoint."""
            eurosat_train.main()


        if __name__ == "__main__":
            main()