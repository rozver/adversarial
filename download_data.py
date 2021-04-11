import gdown
import zipfile
import os

# Download zip file from Google Drive
drive_url = 'https://drive.google.com/uc?id=13vhvMGc56CdQx7DrGtVVi86GFBm5Zu2J'
output_file = 'adversarial_data.zip'
gdown.download(drive_url, output_file, quiet=False)

# Extract the zip file into the current directory
with zipfile.ZipFile(output_file, 'r') as file:
    file.extractall('')

# Remove the zip file
os.remove(output_file)
