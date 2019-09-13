#! python

import xml.etree.ElementTree
from glob import glob

import pandas as pd

import config

bag = [
    # (filename, tag)
]


def get_stats():
    for annotation_file in glob('{}/*.xml'.format(config.ANNOTATIONS_DIR)):
        root = xml.etree.ElementTree.parse(annotation_file).getroot()
        for each_object in root.findall('object'):
            name = each_object.find('name').text
            bag.append((annotation_file, name))

    df = pd.DataFrame(bag, columns=['filename', 'tag'])
    print('Loc: {}'.format(config.ANNOTATIONS_DIR))
    print('Total Number of files processed: {}'.format(len(df.filename.unique())))
    return df.tag.value_counts().to_frame().to_html(classes='table table-striped')


if __name__ == '__main__':
    print(get_stats())
