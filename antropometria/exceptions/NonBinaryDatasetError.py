class NonBinaryDatasetError(Exception):
    def __init__(self, number_of_classes: int):
        self.number_of_classes = number_of_classes
        super().__init__()

    def __str__(self):
        return f'Expected dataset to have 2 classes, intead had {self.number_of_classes}'
