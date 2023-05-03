import os
from ftplib import FTP

ftp_username = 'jmoore003'
ftp_password = '(f_0u6{{7OYt'


def download_files_from_dir(ftp, path, output_dir):
    current_dir = ftp.pwd()

    ftp.cwd(path)
    items = ftp.nlst()

    for item in items:
        item_type = ftp.sendcmd(f'MLST {item}').split(';')[0].split('=')[-1].strip()
        if item_type == 'dir':
            download_files_from_dir(ftp, item, output_dir)
        elif item.endswith('.csv'):
            local_file_path = os.path.join(output_dir, item)
            print(f'Downloading {item}...')
            with open(local_file_path, 'wb') as local_file:
                ftp.retrbinary(f'RETR {item}', local_file.write)
            print(f'Successfully downloaded {item}.')

    ftp.cwd(current_dir)

# Set up FTP connection
ftp = FTP('ftp.ceda.ac.uk')
ftp.login(user=ftp_username, passwd=ftp_password)

# Navigate to the target directory
target_dir = '/badc/ukmo-midas-open/data/uk-daily-weather-obs/dataset-version-202207/'

# Create a local directory to store the downloaded CSV files
output_dir = 'historical_station_data'
os.makedirs(output_dir, exist_ok=True)

# Download CSV files from the target directory and its subdirectories
download_files_from_dir(ftp, target_dir, output_dir)

# Close FTP connection
ftp.quit()

print('Download process completed.')