import re
from pathlib import Path
import collections
import numbers
import sklearn.utils.multiclass as skutils
import pandas as pd
import numpy as np
import platform
import shutil
import os
import tempfile
import random
from ml101.data import Types, TAR


#==============================================
#   System / OS
#==============================================
def is_linux() -> bool:
    return platform.system() == 'Linux'


#==============================================
#   File Helpers
#==============================================
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


def listdir(dirpath) -> list:
    return [file for file in Path(dirpath).iterdir() if file.is_file()]


def copy(srcfile:Path, dst:Path, symbolic=False) -> Path:
    srcfile = Path(srcfile)
    dst = Path(dst)
    if symbolic and is_linux():
        dstfile = dst / srcfile.name if dst.is_dir() else dst
        if srcfile.is_symlink():
            srcfile = (srcfile.parent / os.readlink(srcfile)).resolve()
        rel_path_src = os.path.relpath(srcfile, dst)
        if (dstfile.is_symlink() or dstfile.exists()): dstfile.unlink()
        dstfile.symlink_to(rel_path_src, target_is_directory=not is_linux())
    else:
        dstfile = shutil.copy(srcfile, dst)

    return Path(dstfile)


def get_temp_name(prefix:str=None) -> str:
    if prefix:
        return prefix + str(random.randint(0, 999))
    return next(tempfile._get_candidate_names())


#==============================================
#   Data Conversion
#==============================================
def round_container(src, ndigits=4):
    if isinstance(src, dict):
        return type(src)((key, round_container(value, ndigits)) for key, value in src.items())
    if isinstance(src, str) is False and isinstance(src, collections.Container):
        return type(src)(round_container(value, ndigits) for value in src)
    if isinstance(src, numbers.Number):
        return round(src, ndigits)
    return src


def reverse_dict(src:dict) -> dict:
    return {value:key for key, value in src.items()}


def normalize_matrix(mat: TAR, normalize: str='all'):
    mat_numpy = Types.check_array(mat)

    with np.errstate(all='ignore'):
        normalized_mat = None
        sum_mat = None
        if normalize == 'row':
            sum_mat = mat_numpy.sum(axis=1, keepdims=True)
            normalized_mat = mat_numpy / sum_mat
            sum_mat = np.tile(sum_mat, (1, mat_numpy.shape[1]))
        elif normalize == 'column':
            sum_mat = mat_numpy.sum(axis=0, keepdims=True)
            normalized_mat = mat_numpy / sum_mat
            sum_mat = np.tile(sum_mat, (mat_numpy.shape[0], 1))
        elif normalize == 'all':
            sum_mat = mat_numpy.sum()
            normalized_mat = mat_numpy / sum_mat
            sum_mat = np.tile(sum_mat, mat_numpy.shape)
        normalized_mat = np.nan_to_num(normalized_mat)
    return normalized_mat, sum_mat


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


def rearrange(containers):
    return [item for group in zip(*containers) for item in group]


#==============================================
#   Dataset
#==============================================
def build_idx2labels(*labels: list) -> dict:
    unique = skutils.unique_labels(*labels)
    return {idx: label for idx, label in enumerate(unique)}


#==============================================
#   Class
#==============================================
def update_kwargs(default_set: dict, new_set: dict) -> dict:
    """Update deafult_set with common keys-values from new_set

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