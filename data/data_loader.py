import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from typing import Tuple
from .market_simulator import MarketSimulator, MarketConfig

class TradingDataset(Dataset):
    """Dataset for trading scenarios"""
    
    def __init__(self, simulator: MarketSimulator, n_samples: int):
        self.simulator = simulator
        self.n_samples = n_samples
        
        # Pre-generate all data
        self.prices, self.holdings = simulator.generate_batch(n_samples)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.prices = self.prices.cuda()
            self.holdings = self.holdings.cuda()
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.prices[idx], self.holdings[idx]

def create_data_loader(config: MarketConfig, n_samples: int, batch_size: int) -> TorchDataLoader:
    """Create a data loader for training"""
    simulator = MarketSimulator(config)
    dataset = TradingDataset(simulator, n_samples)
    return TorchDataLoader(dataset, batch_size=batch_size, shuffle=True)