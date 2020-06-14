from argparse import Namespace
from config import get_config, get_config_fields, special_args
from collections import OrderedDict
import zlib

import numpy as np
import torch

fields = get_config_fields()

sep = '_'

def get_args(config):
    blank_namespace = Namespace()
    args = {k: v for k, v in config.__dict__.items() if (k not in blank_namespace) and (k not in special_args)}
    return OrderedDict(sorted(args.items()))


def concatenate_strings(strings, separator, nonallowed_chars, add_hash=True):
    MAX_LENGTH = 3
    nonallowed_chars.append(separator)
    def remove_bad(string):
        for char in nonallowed_chars:
            string = string.replace(char, '')
        return string
    longstring = separator.join(strings)
    stringhash = zlib.adler32(bytes(longstring, 'utf-8'))   # ensure uniqueness using hash from before strings were truncated
    strings = [remove_bad(string) for string in strings]  # remove non-allowed characters
    strings = [string[:MAX_LENGTH] for string in strings] # truncate long strings
    if add_hash:
        strings.append(str(stringhash))
    return separator.join(strings)


def dict2namespace(config_dict):
    namespace = Namespace()
    for field, value in config_dict.items():
        setattr(namespace, field, value)
    return namespace


def get_model_name(config, config_type='Namespace'):
    if config_type == 'dict':
        config = dict2namespace(config)
    assert type(config) == Namespace
    # make name assuming config is a Namespace
    args = get_args(config)
    string_vals = [str(v) for v in args.values()]
    return concatenate_strings(string_vals, sep, ['/'])


def name_get_value(name, field):
    assert field in fields
    index = fields.index(field)
    return name.split(sep)[index]


def name_replace_value(name, field, new_value):
    # doesn't change hash so typically returns invalid name
    assert field in fields or field == 'hash'
    index = -1 if field == 'hash' else fields.index(field)
    vals = name.split(sep)
    vals[index] = str(new_value)
    return concatenate_strings(vals, sep, ['/'], add_hash=False)


def save_config(config, path, config_type='Namespace'):
    if config_type == 'dict':
        config = dict2namespace(config)
    assert type(config) == Namespace
    args = get_args(config)
    with open(path, 'w') as f:
        for field, val in args.items():
            f.write(field)
            f.write(' ')
            f.write(str(val))
            f.write('\n')


def read_config(path):
    """
    returns config as dictionary with all values as strings
    """
    config_dict = {}
    for line in open(path, 'r').read().splitlines():
        field, value = line.split(' ')
        config_dict[field] = value
    return config_dict


def config_different_fields(config1, config2):
    differences = []
    for field, v1 in config1.items():
        assert field in config2, "fields do not match between configs"
        if v1 != config2[field]:
            differences.append(field)
    return differences
