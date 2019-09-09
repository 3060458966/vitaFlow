"""
Module for working with xml files(reading & wri.
"""

import json
import os
import xml.dom.minidom

import xmltodict

try:
    from . import config
except ImportError:
    import config


def single_annotated_box_to_xml(annotation):
    _object_xml_format = dict({'name': annotation['tag'],
                               'pose': 'Unspecified',
                               'truncated': '0',
                               'difficult': '0',
                               'bndbox': {
                                   'xmin': annotation['x'],
                                   'ymin': annotation['y'],
                                   'xmax': annotation['x'] + annotation['width'],
                                   'ymax': annotation['y'] + annotation['height']}
                               })
    return _object_xml_format


def annotated_boxes_to_xml(annotations):
    bag = []
    for each in annotations:
        bag.append(single_annotated_box_to_xml(each))
    return bag


def validate_tags_and_regions(form_data):
    sent_info = json.loads(form_data['sendInfo'])
    # sent_info.keys() => dict_keys(['url', 'folder', 'id', 'width', 'height', 'annotations'])
    print('Generating XML file for {}'.format(sent_info['id']))
    default_xml_format = {
        'annotation': {
            'folder': sent_info['folder'],
            'filename': sent_info['id'],
            'path': sent_info['url'],
            'source': {
                'database': 'Unknown'
            },
            'size_part': {
                'width': sent_info['width'],
                'height': sent_info['height'],
                'depth': '3'},
            'segmented': '0',
            'object': annotated_boxes_to_xml(sent_info['annotations'])
        }
    }
    xml_string = xmltodict.unparse(default_xml_format)
    xml_string = xml.dom.minidom.parseString(xml_string)
    pretty_xml_as_string = xml_string.toprettyxml()
    # print(pretty_xml_as_string)
    file_name = os.path.join(config.ANNOTATIONS_DIR, sent_info['id'].rsplit('.')[-2] + '.xml')
    open(file_name, 'w').write(pretty_xml_as_string)
    print('Wrote xml data to file {}'.format(file_name))


def xml_to_single_annotated_box(xml_annotation):
    return {
        'tag': xml_annotation['name'],
        'x': float(xml_annotation['bndbox']['xmin']),
        'y': float(xml_annotation['bndbox']['ymin']),
        'width': round(float(xml_annotation['bndbox']['xmax']) - float(xml_annotation['bndbox']['xmin']),),
        'hight': round(float(xml_annotation['bndbox']['ymax']) - float(xml_annotation['bndbox']['ymin'])),
    }


def xml_to_annotated_boxes(xml_annotations):
    bag = []
    for each in xml_annotations:
        bag.append(xml_to_single_annotated_box(each))
    return bag


def read_xml_annotated_file(filename):
    xml_string = open(filename).read()
    data = xmltodict.parse(xml_string)
    data = dict(data)
    data['annotation'] = dict(data['annotation'])
    data['annotation']['source'] = dict(data['annotation']['source'])
    data['annotation']['size_part'] = dict(data['annotation']['size_part'])
    data['annotation']['object'] = list(map(dict, data['annotation']['object']))
    for i in range(len(data['annotation']['object'])):
        data['annotation']['object'][i]['bndbox'] = dict(data['annotation']['object'][i]['bndbox'])
    return data
