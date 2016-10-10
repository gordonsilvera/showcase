# library inputs
import numpy as np
import pandas as pd
import datetime
import urllib
import cnfg

# load configuration settings
settings = cnfg.load('Anaconda/github/sandbox/.gms_config')

# load all businesses within a given zipcode (https://dev.socrata.com/foundry/data.cityofnewyork.us/<nyc_key>)
def loadZip(zipcode):
    #DOC: 
    query = "https://data.cityofnewyork.us/resource/" + settings['nyc_key'] + ".json?address_zip=" + str(zipcode) 
    return pd.read_json(query)

# output zipcode businesses to data frame
raw_data = loadZip(11217)

raw_data[0:5]
#raw_data.shape
#raw_data.describe()
#raw_data.head()
#raw_data.license_category.unique()
#raw_data.license_category.value_counts()



