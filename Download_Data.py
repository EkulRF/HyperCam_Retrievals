import requests

the_url = 'https://liveuclac-my.sharepoint.com/:f:/g/personal/ucfalri_ucl_ac_uk/Eo1MgmImegpDmlRtRjfLTc4BQUfw73dds0ha8Qk5u4kPZw?download=1'
response = requests.get(the_url)

with open('Test_Data/test_me.pdf', 'wb') as fp:
    fp.write(response.content)
