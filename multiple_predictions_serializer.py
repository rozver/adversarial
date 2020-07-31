import torch
import os
import sys
import pandas as pd
from get_image_prediction import predict


class PredictionsDictionary:
    def __init__(self, folder_location):
        self.dictionary = {}
        self.current_folder_location = folder_location
        self.predict_recursive()

    def predict_recursive(self):
        folder_location = self.current_folder_location
        for entity in os.listdir(folder_location):
            entity_location = os.path.join(folder_location, entity)
            if entity.endswith(('.png', '.jpg', '.jpeg')):
                predicted_class = torch.argmax(predict(entity_location)).item()
                self.dictionary[entity_location] = predicted_class
            elif os.path.isdir(entity_location):
                self.current_folder_location = entity_location
                self.predict_recursive()


def save_dictionary_as_csv(predictions_dictionary, csv_file):
    with open(csv_file, 'w') as f:
        (pd.DataFrame.from_dict(data=predictions_dictionary, orient='index')
         .to_csv(f, header=False))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        location = sys.argv[1]
        if os.path.isdir(location):
            print('Starting to make predictions...')
            dictionary = PredictionsDictionary(location).dictionary
            print('Finished')

            print('Starting serialization...')
            csv_file_location = os.path.join(location, 'labels.csv')
            save_dictionary_as_csv(dictionary, csv_file_location)
            print('Finished')
        else:
            print('Invalid folder location!')
    else:
        print('Enter path of the folder in which the images are stored!')
