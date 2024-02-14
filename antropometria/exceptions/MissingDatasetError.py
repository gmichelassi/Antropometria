class MissingDatasetError(Exception):
    def __init__(self, folder: str, name: str):
        self.folder = folder
        self.name = name
        super().__init__()

    def __str__(self):
        return f'File not found for arguments folder={self.folder}, name={self.name}'
