import torch
import os
import sys
import pandas as pd
from get_image_prediction import predict


def predict_recursive(current_folder_location, predictions_dictionary):
    for entity in os.listdir(current_folder_location):
        entity_location = os.path.join(current_folder_location, entity)
        if entity.endswith(('.png', '.jpg', '.jpeg')):
            predicted_class = torch.argmax(predict(entity_location)).item()
            predictions_dictionary[entity_location] = predicted_class
        elif os.path.isdir(entity_location):
            predict_recursive(entity_location, predictions_dictionary)


def save_dictionary_as_csv(predictions_dictionary, csv_file):
    with open(csv_file, 'w') as f:
        (pd.DataFrame.from_dict(data=predictions_dictionary, orient='index')
         .to_csv(f, header=False))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        location = sys.argv[1]
        if os.path.isdir(location):
            print('Starting to make predictions...')
            dictionary = {}
            predict_recursive(location, dictionary)
            print('Finished')

            print('Starting serialization...')
            csv_file_location = os.path.join(location, 'labels.csv')
            save_dictionary_as_csv(dictionary, csv_file_location)
            print('Finished')
        else:
            print('Invalid folder location!')
    else:
        print('Enter path of the folder in which the images are stored!')
