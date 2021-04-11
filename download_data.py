import requests
import zipfile
import os

# Download zip file from Google Drive
drive_url = 'https://drive.google.com/uc?id=13vhvMGc56CdQx7DrGtVVi86GFBm5Zu2J'
output_location = 'adversarial_data.zip'

print('Downloading data...')
r = requests.get(drive_url, allow_redirects=True)
with open(output_location, 'wb') as output_file:
    output_file.write(r.content)
print('Finished!')

# Extract the zip file into the current directory
print('Unzipping file...')
with zipfile.ZipFile(output_location, 'r') as zip_file:
    zip_file.extractall('')
print('Finished!')

# Remove the zip file
print('Removing the zip file...')
os.remove(output_location)
print('Finished!')
