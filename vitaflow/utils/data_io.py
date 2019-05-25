import os
import sys
import tarfile
import zipfile
import collections
import requests
import numpy as np
import tensorflow as tf


from six.moves import urllib


def maybe_create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return True
    return False


def maybe_download(urls, path, filenames=None, extract=False):
    """Downloads a set of files.

    Args:
        urls: A (list of) urls to download files.
        path (str): The destination path to save the files.
        filenames: A (list of) strings of the file names. If given,
            must have the same length with :attr:`urls`. If `None`,
            filenames are extracted from :attr:`urls`.
        extract (bool): Whether to extract compressed files.

    Returns:
        A list of paths to the downloaded files.
    """
    maybe_create_dir(path)

    if not isinstance(urls, (list, tuple)):
        urls = [urls]
    if filenames is not None:
        if not isinstance(filenames, (list, tuple)):
            filenames = [filenames]
        if len(urls) != len(filenames):
            raise ValueError(
                '`filenames` must have the same number of elements as `urls`.')

    result = []
    for i, url in enumerate(urls):
        if filenames is not None:
            filename = filenames[i]
        elif 'drive.google.com' in url:
            filename = _extract_google_drive_file_id(url)
        else:
            filename = url.split('/')[-1]
            # If downloading from GitHub, remove suffix ?raw=True
            # from local filename
            if filename.endswith("?raw=true"):
                filename = filename[:-9]

        filepath = os.path.join(path, filename)
        result.append(filepath)

        if not tf.gfile.Exists(filepath):
            if 'drive.google.com' in url:
                filepath = _download_from_google_drive(url, filename, path)
            else:
                filepath = _download(url, filename, path)

        if extract:
            tf.logging.info('Extract %s', filepath)
            if tarfile.is_tarfile(filepath):
                tarfile.open(filepath, 'r').extractall(path)
            elif zipfile.is_zipfile(filepath):
                with zipfile.ZipFile(filepath) as zfile:
                    zfile.extractall(path)
            else:
                tf.logging.info("Unknown compression type. Only .tar.gz, "
                                ".tar.bz2, .tar, and .zip are supported")

    return result


def _download(url, filename, path):
    def _progress(count, block_size, total_size):
        percent = float(count * block_size) / float(total_size) * 100.
        # pylint: disable=cell-var-from-loop
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (filename, percent))
        sys.stdout.flush()

    filepath = os.path.join(path, filename)
    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print(filepath)
    statinfo = os.stat(filepath)
    print('Successfully downloaded {} {} bytes.'.format(
        filename, statinfo.st_size))

    return filepath


def _extract_google_drive_file_id(url):
    # id is between `/d/` and '/'
    #     url_suffix = url[url.find('/d/')+3:]
    #     file_id = url_suffix[:url_suffix.find('/')]
    #     print(file_id)
    file_id = url.split("=")[-1]
    return file_id


def _download_from_google_drive(url, filename, path):
    """Adapted from `https://github.com/saurabhshri/gdrive-downloader`
    """
    print(url)

    def _get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    file_id = _extract_google_drive_file_id(url)

    gurl = "https://docs.google.com/uc?export=download"
    sess = requests.Session()
    response = sess.get(gurl, params={'id': file_id}, stream=True)
    print(response)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = sess.get(gurl, params=params, stream=True)

    filepath = os.path.join(path, filename)
    CHUNK_SIZE = 32768
    with tf.gfile.GFile(filepath, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

    print('Successfully downloaded {}.'.format(filename))

    return filepath
