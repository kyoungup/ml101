import re
from pathlib import Path
import collections
import numbers
import sklearn.utils.multiclass as skutils
import numpy as np
import platform
import shutil
import os


def confirm_path(fullpath) -> Path:
    path = None
    filename = None
    if fullpath:
        fullpath = Path(fullpath)
        if fullpath.suffix != '':   # check if it is a file
            filename = fullpath.name
            path = fullpath.parent
        else:
            path = fullpath
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


def update_kwargs(default_set: dict, new_set: dict) -> dict:
    """Update only existing key-values of default set with a new set

    Args:
        default_set (dict): [default kwargs]
        new_set (dict): [a set of new values]

    Returns:
        dict: [description]
    """
    kwargs = dict()
    for arg in default_set:
        kwargs[arg] = new_set[arg] if arg in new_set else default_set[arg]
    return kwargs


def convert4json(container):
    if isinstance(container, dict):
        for key, value in container.items():
            container[key] = convert4json(value)
    elif isinstance(container, list):
        for idx, value in enumerate(container):
            container[idx] = convert4json(value)
    elif isinstance(container, np.ndarray):
        return container.tolist()
    elif isinstance(container, Path):
        return str(container)
    return container


def listdir(dirpath) -> list:
    return [file for file in Path(dirpath).iterdir() if file.is_file()]


def is_linux() -> bool:
    return platform.system() == 'Linux'


def copy(srcfile:Path, dst:Path, symbolic=False) -> Path:
    srcfile = Path(srcfile)
    dst = Path(dst)
    if symbolic and is_linux():
        dstfile = dst / srcfile.name if dst.is_dir() else dst
        if srcfile.is_symlink():
            srcfile = (srcfile / os.readlink(srcfile)).resolve()
        rel_path_src = os.path.relpath(srcfile, dst)
        if (dstfile.is_symlink() or dstfile.exists()): dstfile.unlink()
        dstfile.symlink_to(rel_path_src, target_is_directory=not is_linux())
    else:
        dstfile = shutil.copy(srcfile, dst)

    return Path(dstfile)