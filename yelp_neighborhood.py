
# Github Doc: https://github.com/Yelp/yelp-python
from yelp.client import Client
from yelp.oauth1_authenticator import Oauth1Authenticator
import cnfg


settings = cnfg.load('data-scrubber/python/.gms_config')

# Yelp Authentication
auth = Oauth1Authenticator(
    consumer_key = settings['consumer_key'],
    consumer_secret = settings['consumer_secret'],
    token = settings['token'],
    token_secret = settings['token_secret']
)

client = Client(auth)



# search parameter
params = {
    'term': 'food',
    'lang': 'en'
}
    
response = client.search('Fort Greene, Brooklyn', **params)


# Parse Search Results
def pullHood(hood):
    response = client.search(hood, **params)
    
    for i in range(len(response.businesses)):
        #business_id = response.businesses[i].id
        business_list = {
            'id': response.businesses[i].id, 
            'is_claimed': response.businesses[i].is_claimed,
            'is_closed': response.businesses[i].is_closed,
            'name': response.businesses[i].name,
            'image_url': response.businesses[i].image_url,
            'url': response.businesses[i].url,
            'phone': response.businesses[i].phone,
            'display_phone': response.businesses[i].display_phone,
            'review_count': response.businesses[i].review_count,
            'categories': response.businesses[i].categories,
            'rating': response.businesses[i].rating,
            'snippet_text': response.businesses[i].snippet_text,
            'snippet_image_url': response.businesses[i].snippet_image_url,
        
            'address': response.businesses[i].location.address,
            'display_address': response.businesses[i].location.display_address,
            'city': response.businesses[i].location.city,
            'state_code': response.businesses[i].location.state_code,
            'postal_code': response.businesses[i].location.postal_code,
            'country_code': response.businesses[i].location.country_code,
            'cross_streets': response.businesses[i].location.cross_streets,
            'neighborhoods': response.businesses[i].location.neighborhoods,
            'latitude': response.businesses[i].location.coordinate.latitude,
            'longitude': response.businesses[i].location.coordinate.longitude
            }
        business_dict[i] = business_list

    return business_dict


# run pullHood: creates dictionary of all
outHood = pullHood('Fort Greene, Brooklyn')


# upload to a pandas  data frame
import pandas as pd

df_business = pd.DataFrame(outHood).T
df_business.head()
