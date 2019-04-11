# https://github.com/google/fonts

import base64
import os
import random
import shutil
import string
import time

import requests
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# import xml.etree.ElementTree as ET

try:
    from vitaflow.image_synthesizer.config import ALL_MERCHANTS_ADDR, ALL_MERCHANTS_NAMES, ALL_LINE_ITEMS, OUT_DIR
except ImportError:
    try:
        from config import ALL_MERCHANTS_ADDR, ALL_MERCHANTS_NAMES, ALL_LINE_ITEMS, OUT_DIR
    except ImportError:
        print('Not able to find config - variables!!')
        os._exit(1)


def get_random_merchant_name():
    return random.choice(ALL_MERCHANTS_NAMES)


def get_random_line_item():
    return random.choice(ALL_LINE_ITEMS)


def get_random_merchant_address():
    return random.choice(ALL_MERCHANTS_ADDR)


def get_random_string(max_num_chars=15):
    num_chars = random.randint(5, max_num_chars)
    # digits = "".join( [random.choice(string.digits) for i in range(8)] )
    chars = "".join([random.choice(string.ascii_letters) for i in range(num_chars)])
    return chars


def get_random_number(min_num_chars=1, max_num_chars=5):
    num_chars = random.randint(min_num_chars, max_num_chars)
    digits = "".join([random.choice(string.digits) for i in range(num_chars + 2)])
    x = float(digits) / 100.0
    return "{:5.2f}".format(x)


def strTimeProp(start, end, format, prop):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """

    stime = time.mktime(time.strptime(start, format))
    etime = time.mktime(time.strptime(end, format))

    ptime = stime + prop * (etime - stime)

    return time.strftime(format, time.localtime(ptime))


def get_random_date(start, end, prop):
    text = strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)
    date, time, am_pm = text.split(" ")
    return date, time, am_pm


def read_from_git(url):
    url = "https://github.com/google/fonts/blob/master/apache/roboto/Roboto-Bold.ttf"
    req = requests.get(url)
    if req.status_code == requests.codes.ok:
        req = req.json()  # the response is a JSON
        # req is now a dict with keys: name, encoding, url, size ...
        # and content. But it is encoded with base64.
        content = base64.decodebytes(req['content'])
        return content
    else:
        print('Content was not found.')


def insert_text(draw: ImageDraw,
                x,
                y,
                text,
                color='rgb(0, 0, 0)',
                font_file='fonts/Roboto-Bold.ttf',
                font_size=14):
    text = str(text)
    font = ImageFont.truetype(font_file, size=font_size)
    draw.text((x, y), text, fill=color, font=font)
    return draw


def generate_ann(text_data):
    res_bag = []
    _counter = 0

    # merchant
    _merchant = text_data.splitlines()[0]
    _m1 = 0
    _m2 = len(_merchant)
    _counter += 1
    res_bag.append('T{}\tMerchant {} {}\t{}'.format(_counter, _m1, _m2, _merchant))

    # Date
    _d1 = text_data.find('Date')
    _d2 = _d1 + len('Date : 05/08/2018')
    _date = text_data[_d1:_d2]
    _counter += 1
    res_bag.append('T{}\tDate {} {}\t{}'.format(_counter, _d1, _d2, _date))

    # LineItems
    _t1 = text_data.find('Items  Prices\n') + len('Items  Prices\n')
    _t2 = text_data.find('Tax') - 1
    _line1 = _t1
    for line in text_data[_t1:_t2].splitlines():
        _line2 = _line1 + len(line)
        _counter += 1
        res_bag.append('T{}\tLineItem {} {}\t{}'.format(_counter, _line1, _line2, line))
        _line1 = _line1 + len(line) + 1

    # Total
    _total = text_data.splitlines()[-1].strip()
    _t1 = text_data.find(_total)
    _t2 = _t1 + len(_total)
    # _t2 = len(text_data)
    _counter += 1
    res_bag.append('T{}\tTotal {} {}\t{}'.format(_counter, _t1, _t2, _total))

    return '\n'.join(res_bag)


def create_naive_receipt(file_path):
    # create Image object with the input image
    # TODO: Generated a proper text receipt & from it generated receipt
    # TODO: Generate Header, Body & Footer - as seperate regions and image-concated them
    # TODO: Convert this to a class model
    receipt_text = []
    _number_of_line_items = random.choice(range(5, 15))
    _image_size = (360, 250 + _number_of_line_items * 15)
    # one table center - column
    one_table_col1 = int(_image_size[0] * 0.25)
    one_table_col2 = int(_image_size[0] * 0.15)
    one_table_col21 = int(_image_size[0] * 0.10)
    one_table_col22 = int(_image_size[0] * 0.55)
    # two table columns
    two_table_col1 = int(_image_size[0] * 0.15)
    two_table_col2 = int(_image_size[0] * 0.65)
    #
    footer_col1 = int(_image_size[0] * 0.55)
    footer_col2 = int(_image_size[0] * 0.45)
    # print(f'Image Size: _image_size')
    _default_font = 14

    image = Image.new(mode="RGB", size=_image_size, color=(255, 255, 255))
    # initialise the drawing context with
    # the image object as background
    draw = ImageDraw.Draw(image)

    _merchant_tag = get_random_merchant_name().center(15)
    _merchant_addr = get_random_merchant_address().center(50)
    _merchant_addr2 = '{}-{}-{}'.format(get_random_number(3, 3).split('.')[0],
                                        get_random_number(3, 3).split('.')[0],
                                        get_random_number(4, 4).split('.')[0]).center(50)
    _date = random.random()
    date, _, _ = get_random_date("1/1/2015 1:30 PM", "1/1/2019 4:50 AM", _date)
    date = "Date : " + str(date)
    _invoice_text = "Invoice : " + str(get_random_number(min_num_chars=4, max_num_chars=6).split('.')[0])

    receipt_text = [
        _merchant_tag.strip(),
        _merchant_addr.strip(),
        _merchant_addr2.strip(),
        _invoice_text.strip() + ' ' + date,
        "Items  Prices",
    ]
    ########################################## HEADER
    # first line
    draw = insert_text(draw=draw, x=one_table_col1, y=30, text=_merchant_tag, font_size=25)
    draw = insert_text(draw=draw, x=one_table_col2, y=65, text=_merchant_addr, font_size=14)
    draw = insert_text(draw=draw, x=one_table_col2, y=80, text=_merchant_addr2, font_size=14)

    # invoice - date line
    draw = insert_text(draw=draw, x=one_table_col21, y=100, text=_invoice_text, font_size=14)
    draw = insert_text(draw=draw, x=one_table_col22, y=100, text=date, font_size=14)

    # table header
    draw = insert_text(draw=draw, x=two_table_col1, y=130, text="Items", font_size=14)
    draw = insert_text(draw=draw, x=two_table_col2, y=130, text="Prices", font_size=14)

    ########################################## BODY
    total = 0
    total_y = -1
    _min_line_width = 15
    for i in range(1, _number_of_line_items):
        # item
        item_text = get_random_line_item()
        draw = insert_text(draw=draw, x=two_table_col1, y=140 + (i * _min_line_width), text=item_text, font_size=14)

        # item - price
        item_price = '%5s' % float(get_random_number(max_num_chars=3))
        draw = insert_text(draw=draw, x=two_table_col2, y=140 + (i * _min_line_width), text=item_price, font_size=14)
        receipt_text.append('{}  {}'.format(item_text.strip(), item_price).strip())

        total = total + float(item_price)

    total_y = 140 + (i * _min_line_width)

    total_y += 10
    # draw = insert_text(draw=draw, x=20, y=total_y, text="-" * 45 , font_size=14)
    total_y += 20
    _text = "Tax : " + str(0.15)
    receipt_text.append(_text)
    draw = insert_text(draw=draw, x=footer_col1, y=total_y, text=_text, font_size=14)
    total_y += 20
    _text = "Sub Total : " + str(round(total, 2))
    receipt_text.append(_text)
    draw = insert_text(draw=draw, x=footer_col1, y=total_y, text=_text, font_size=14)
    total_y += 20
    _tag_total = random.choice([
        "        Total: ",
        "   Amount Due: ",
        " Total Amount: ",
        " Total To Pay:",
        "Total Charges:",
    ])
    _text = _tag_total + str(round(total + total * 0.15, 2))
    receipt_text.append(_text)
    draw = insert_text(draw=draw, x=footer_col2 - len(_tag_total), y=total_y, text=_text, font_size=14 + 4)

    # write to a file
    fp = open(file_path + '.txt', 'w')
    fp_ann = open(file_path + '.ann', 'w')
    text_data = '\n'.join(receipt_text)
    try:
        fp.write(text_data)
        fp_ann.write(generate_ann(text_data))
    except UnicodeEncodeError:
        import pdb
        pdb.set_trace()
    # save image
    image.save(file_path, "PNG")


# def replicate_xml(out_file_path, image_store_path="images", in_file_pathh="0.xml"):
#     tree = ET.parse(in_file_pathh)
#     root = tree.getroot()
#     for elem in root.iter('filename'):
#         elem.text = out_file_path.split("/")[-1]
#     for elem in root.iter('path'):
#         elem.text = image_store_path + "/" + out_file_path.split("/")[-1]
#     tree.write(out_file_path, encoding='utf-8', xml_declaration=True)


def train():
    number_files = 200
    os.makedirs(OUT_DIR + "/train/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/train/" + str(i) + ".png")
        # replicate_xml(out_file_path=OUT_DIR + "/train/" + str(i) + ".xml")


def test():
    number_files = 50
    os.makedirs(OUT_DIR + "/test/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/test/" + str(i) + ".png")
        # replicate_xml(out_file_path=OUT_DIR + "/test/" + str(i) + ".xml")


def eval():
    number_files = 25
    os.makedirs(OUT_DIR + "/val/")

    for i in tqdm(range(number_files)):
        create_naive_receipt(OUT_DIR + "/val/" + str(i) + ".png")
        # replicate_xml(out_file_path=OUT_DIR + "/eval/" + str(i) + ".xml")


def main():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    train()
    test()
    eval()


if __name__ == '__main__':
    main()
