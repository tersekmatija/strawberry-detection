import torch

from datasets.strawberrydi import StrawDIDataset

def get_loader(dataset_name, split, dataset_dir, batch_size, transforms=None, num_workers=0, pin_memory=False, shuffle=False):

    if dataset_name.lower() == "strawdi":
        dataset = StrawDIDataset(split=split, root=dataset_dir, transforms=transforms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=StrawDIDataset.collate_fn, shuffle=shuffle, pin_memory=pin_memory)
    else:
        raise RuntimeError(f"Dataset {dataset_name} not implemented!")

    return loader