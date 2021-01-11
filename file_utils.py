import datetime
import os


def get_current_time():
    return datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')


def validate_save_file_location(location):
    if not location.endswith('.pt'):
        raise ValueError('The files in which the object are going to be serialized must have a .pt extension!')

    save_file_parent_directory = os.path.dirname(location)
    if not os.path.exists(save_file_parent_directory):
        os.makedirs(save_file_parent_directory)
