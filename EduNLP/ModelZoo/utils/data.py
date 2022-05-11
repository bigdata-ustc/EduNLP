import json


def load_items(data_path):
    _data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f.readlines():
            _data.append(json.loads(line))
    return _data
