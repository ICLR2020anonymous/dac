from torchvision.datasets import MNIST
# inspired from https://gist.github.com/y0ast/f69966e308e549f013a92dc66debeeb4

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data.unsqueeze(1).float().div(255.)

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, y
