from django.conf import settings
from datetime import datetime
def rename_and_upload_path(instance, filename):
    """

    :param file: request.FILES object
    :return:
    """
    img_dir = settings.IMG_DIR
    now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    filename, file_format = filename.split('.')
    new_file_name = img_dir + filename + "_{}.{}".format(now, file_format)

    return new_file_name