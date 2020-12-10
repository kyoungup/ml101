import re
from pathlib import Path
import collections
import numbers
import sklearn.utils.multiclass as skutils


def confirm_path(path) -> Path:
    filename = None
    if path:
        path = Path(path)
        if path.suffix != '':   # check if it is a file
            path = path.parent
            filename = path.name
    return path, filename


def convert2filename(text: str) -> str:
    text = re.sub(' +', ' ', text)
    text = re.sub('(-| )', '_', text)
    return text


def insert2filename(path, prefix:str=None, suffix:str=None):
    if isinstance(path, Path) is False: path = Path(path)

    filename = path
    if prefix:
        filename = filename.with_name(prefix + filename.stem).with_suffix(filename.suffix)
    if suffix:
        filename = filename.with_name(filename.stem + suffix).with_suffix(filename.suffix)
    return filename


def round_container(src, ndigits=4):
    if isinstance(src, dict):
        return type(src)((key, round_container(value, ndigits)) for key, value in src.items())
    if isinstance(src, collections.Container):
        return type(src)(round_container(value, ndigits) for value in src)
    if isinstance(src, numbers.Number):
        return round(src, ndigits)
    return src


def reverse_dict(src:dict) -> dict:
    return {value:key for key, value in src.items()}


def build_idx2labels(*labels: list) -> dict:
    unique = skutils.unique_labels(*labels)
    return {idx: label for idx, label in enumerate(unique)}