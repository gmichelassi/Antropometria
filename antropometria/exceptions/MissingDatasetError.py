class MissingDatasetError(Exception):
    def __init__(self, folder: str, name: str):
        self.folder = folder
        self.name = name
        super().__init__()

    def __str__(self):
        return f'File antropometria/data/{self.folder}/{self.name}.csv not found.'
