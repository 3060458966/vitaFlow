import os
import time

import tqdm

import config
from bin.plugin import pluginApplication


def rename_images(images_path):
    counter = 1
    for file in tqdm.tqdm(os.listdir(images_path), desc='Rename Files'):
        time.sleep(0.05)
        counter += 1
        _rename_file(images_path, file, str(counter).zfill(4))


def _rename_file(images_path, filename, add_prefix=None):
    if filename.startswith('.'):
        return
    newname = _get_file_newname(filename, add_prefix)
    _old_file = os.path.join(images_path, filename)
    _new_file = os.path.join(images_path, newname)
    # print('Rename file \n\t{} to \n\t{}'.format(_old_file, _new_file))
    os.rename(
        _old_file,
        _new_file
    )
    # print('Renamed `{}` to `{}`'.format(filename, newname))


def _get_file_newname(filename, add_prefix=None):
    base_filename, file_ext = filename.rsplit('.', 1)
    base_filename = base_filename.lower().strip().replace(' ', '_')
    base_filename = ''.join([_ for _ in base_filename if _ in '_1234567890qwertyuiopasdfghjklzxcvbnm'])
    base_filename = base_filename[:5] if len(base_filename) > 5 else base_filename
    if add_prefix:
        base_filename = '{}_{}'.format(add_prefix, base_filename)
    new_filename = '{}.{}'.format(base_filename, file_ext)
    return new_filename


class filesProcessing(pluginApplication):
    def inputs(self, images_path=None):
        if images_path:
            self.source = images_path
        else:
            self.source = config.IMAGE_ROOT_DIR

    def run(self):
        rename_images(self.source)




if __name__ == '__main__':
    # rename_images(config.IMAGE_ROOT_DIR)
    fp = filesProcessing()
    fp.quick_run()
