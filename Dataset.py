import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from utils import sample_mask

class Cresci15(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']


    def sample_mask(self, idx, l):
        """Create mask."""
        mask = torch.zeros(l)
        mask[idx] = 1
        return torch.as_tensor(mask, dtype=torch.bool)


    def process(self):
        # Read data into huge `Data` list.

        edge_index = torch.load(self.root + "/edge_index.pt")
        edge_type = torch.load(self.root + "/edge_type.pt")
        label = torch.load(self.root + "/label.pt")
        cat_prop = torch.load(self.root + "/cat_properties_tensor.pt")
        num_prop = torch.load(self.root + "/num_properties_tensor.pt")
        des_tensor = torch.load(self.root + "/des_tensor.pt")
        tweets_tensor = torch.load(self.root + "/tweets_tensor.pt")

        features = torch.cat([cat_prop, num_prop, des_tensor, tweets_tensor], axis=1)
        data = Data(x=features, y =label, edge_index=edge_index)
        data.edge_type = edge_type


        sample_number = len(data.y)

        train_idx = torch.load(self.root + "/train_idx.pt")
        val_idx = torch.load(self.root + "/test_idx.pt")
        test_idx = torch.load(self.root + "/val_idx.pt")

        data.train_mask = self.sample_mask(train_idx, sample_number)
        data.val_mask = self.sample_mask(val_idx, sample_number)
        data.test_mask = self.sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Twibot20(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root


    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]


    @property
    def processed_file_names(self):
        return ['data.pt']



    def process(self):
        device = 'cpu'
        labels = torch.load(self.root + "/label.pt").to(device)
        des_tensor = torch.load(self.root + "/des_tensor.pt").to(device)
        tweets_tensor1 = torch.load(self.root + "/tweets_tensor_p1.pt").to(device)
        tweets_tensor2 = torch.load(self.root + "/tweets_tensor_p2.pt").to(device)
        tweets_tensor = torch.cat([tweets_tensor1, tweets_tensor2], 0)
        num_prop = torch.load(self.root + "/num_prop.pt").to(device)
        category_prop = torch.load(self.root + "/category_prop.pt").to(device)
        edge_index = torch.load(self.root + "/edge_index.pt").to(device)
        edge_type = torch.load(self.root + "/edge_type.pt").to(device)
        x = torch.cat([des_tensor, tweets_tensor, num_prop, category_prop], 1)


        m0 = edge_index[0, :] > 11826
        m1 = edge_index[1, :] > 11826
        m = m0 + m1
        x = x[:11826, :]

        data = Data(x=x, y=labels, edge_index=edge_index)
        data.edge_index = edge_index[:, ~m]
        data.edge_type = edge_type[~m]
        sample_number = len(data.x)

        train_idx = range(8278)
        val_idx = range(8278, 8278 + 2365)
        test_idx = range(8278 + 2365, 8278 + 2365 + 1183)

        data.train_mask = sample_mask(train_idx, sample_number)
        data.val_mask = sample_mask(val_idx, sample_number)
        data.test_mask = sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MGTAB(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.root = root

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def sample_mask(self, idx, l):
        """Create mask."""
        mask = torch.zeros(l)
        mask[idx] = 1
        return torch.as_tensor(mask, dtype=torch.bool)


    def process(self):
        # Read data into huge `Data` list.

        edge_index = torch.load(self.root + "/edge_index.pt")
        edge_index = torch.tensor(edge_index, dtype = torch.int64)
        edge_type = torch.load(self.root + "/edge_type.pt")
        edge_weight = torch.load(self.root + "/edge_weight.pt")
        stance_label = torch.load(self.root + "/labels_stance.pt")
        bot_label = torch.load(self.root + "/labels_bot.pt")

        features = torch.load(self.root + "/features.pt")
        features = features.to(torch.float32)


        data = Data(x=features, edge_index=edge_index)
        data.edge_type = edge_type
        data.edge_weight = edge_weight
        data.y1 = stance_label
        data.y2 = bot_label
        sample_number = len(data.y1)

        train_idx = range(int(0.7*sample_number))
        val_idx = range(int(0.7*sample_number), int(0.9*sample_number))
        test_idx = range(int(0.9*sample_number), int(sample_number))

        data.train_mask = self.sample_mask(train_idx, sample_number)
        data.val_mask = self.sample_mask(val_idx, sample_number)
        data.test_mask = self.sample_mask(test_idx, sample_number)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])