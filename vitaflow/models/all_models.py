# coding=utf-8
# Copyright 2019 The vitaFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Forked from https://github.com/tensorflow/tensor2tensor

"""Imports for problem modules."""

import importlib
import re


MODULES = [
    "vitaflow.models.image.east.east_model"
]

ALL_MODULES = list(MODULES)


def _is_import_err_msg(err_str, module):
    module_pattern = "(.)?".join(["(%s)?" % m for m in module.split(".")])
    return re.match("^No module named (')?%s(')?$" % module_pattern, err_str)


def _handle_errors(errors):
    """Log out and possibly reraise errors during import."""
    if not errors:
        return
    log_all = True  # pylint: disable=unused-variable
    err_msg = "Skipped importing {num_missing} datasets modules."
    print(err_msg.format(num_missing=len(errors)))
    for module, err in errors:
        err_str = str(err)
        if not _is_import_err_msg(err_str, module):
            print("From module %s" % module)
            raise err
        if log_all:
            print("Did not import module: %s; Cause: %s" % (module, err_str))


def import_modules(modules):
    errors = []
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as error:
            errors.append((module, error))
    _handle_errors(errors)
