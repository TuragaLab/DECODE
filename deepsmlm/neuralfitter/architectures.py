from torch.nn.

class EncodeFc:
    def __init__(self):
        pass

    def get_ingridients(self):
        pass


class Data:
    def __init__(self, mode, multi_frame, test_size, train_size=None):
        self.mode = mode
        self.multi_frame
        self.test_size = test_size
        self.train_size = train_size

    def _init_online_dataset(self):
        self.train_size = self.train_size


    def _get_online(self):
        pass

    def get_ingridients(self):
        train_loader = DataLoader(train_data_smlm,
                                  batch_size=hy_par.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        test_loader = DataLoader(test_data_smlm,
                                 batch_size=hy_par.test_size, shuffle=False, num_workers=8, pin_memory=True)
