import torchvision
import numpy as np

def create_sublass(cls):

    class SubLoader(cls):
        def __init__(self, *args, exclude_list=[], **kwargs):
            super(SubLoader, self).__init__(*args, **kwargs)

            if exclude_list == []:
                return

            if self.train:
                labels = np.array(self.targets)
                exclude = np.array(exclude_list).reshape(1, -1)
                mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

                self.data = self.data[mask]
                self.targets = labels[mask].tolist()
            else:
                labels = np.array(self.targets)
                exclude = np.array(exclude_list).reshape(1, -1)
                mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)

                self.data = self.data[mask]
                self.targets = labels[mask].tolist()

    return SubLoader