from torchvision.datasets import EMNIST
# inspired from https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4

class FastEMNIST(EMNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data.unsqueeze(1).float().div(255.).transpose(2,3)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
