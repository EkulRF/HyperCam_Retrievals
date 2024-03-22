import requests

sc_file = 'https://liveuclac-my.sharepoint.com/:u:/r/personal/ucfalri_ucl_ac_uk/Documents/Work/8_TELOPS/Test_Data/2023_TestFlightData/20230612_Flight/WhittlesleyAOI/raw/Mosaic.radiance.sc?download=1'
hdr_file = 'https://liveuclac-my.sharepoint.com/:u:/r/personal/ucfalri_ucl_ac_uk/Documents/Work/8_TELOPS/Test_Data/2023_TestFlightData/20230612_Flight/WhittlesleyAOI/raw/Mosaic.radiance.hdr?download=1'

sc_file = requests.get(sc_file)
hdr_file = requests.get(hdr_file)

with open('Test_Data/Mosaic.radiance.sc', 'wb') as fp:
    fp.write(sc_file.content)

with open('Test_Data/Mosaic.radiance.hdr', 'wb') as fp:
    fp.write(hdr_file.content)
