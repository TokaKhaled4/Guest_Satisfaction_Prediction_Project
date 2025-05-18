import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', None)
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
from textblob import TextBlob
from collections import defaultdict
from datetime import datetime
from sklearn.feature_selection import SelectKBest
import plotly.graph_objects as go
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from imblearn.over_sampling import SMOTE
import random
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import os
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
seed = 0
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

df = pd.read_csv("GuestSatisfactionPredictionMilestone2.csv")

# region DataExploration

print(df['guest_satisfaction'].value_counts())
print(df.shape)
print(df.info())
print(df.columns)
print(df.head())
print(df.tail())
print(df.sample())
print(df.describe(exclude='object'))
print(df.describe(exclude='number'))

# endregion

# region Data PreProcessing

# Checking duplicates on whole data

print(df.duplicated().sum())

# Checking duplicates on ID columns

print(df.duplicated(subset='id').sum())

# region Fixing Numerical Columns

df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)

currency_columns = ['nightly_price', 'price_per_stay', 'extra_people', 'security_deposit', 'cleaning_fee']

for col in currency_columns:
    df[col] = df[col].str.replace(',', '').str.replace('$', '').astype(float)
    
df['zipcode']=df['zipcode'].str.replace("-", "")
df['zipcode']=df['zipcode'].astype(float)

# endregion

# Checking Nulls

print(df.isnull().sum())

# region Handling Nulls in numerical columns    

# Using Mode for categorical columns

mode_fill_columns = ['market', 'host_neighbourhood', 'state', 'neighbourhood', 'host_location','host_response_time','zipcode','host_since','host_is_superhost','host_has_profile_pic','host_identity_verified','host_name']

for col in mode_fill_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
# Using Median for numerical columns

median_fill_columns = [
    'bathrooms', 'bedrooms', 'beds',
    'host_rating', 'host_listings_count', 'host_total_listings_count',
    'host_response_rate'
]
# region plotting
half = math.ceil(len(median_fill_columns) / 2)
columns_split = [median_fill_columns[:half], median_fill_columns[half:]]

for group in columns_split:
    fig, axes = plt.subplots(-(-len(group) // 2), 2, figsize=(12, len(group) * 2))
    axes = axes.flatten()

    for ax, col in zip(axes, group):
        sns.histplot(df[col], kde=True, color="#EE2D6B", edgecolor='black', bins=20, ax=ax)
        ax.set_title(col.replace("_", " ").title())
        ax.set_xlabel('')
        ax.set_ylabel('Frequency')

    for ax in axes[len(group):]:
        ax.remove()

    plt.tight_layout()
    plt.show()
# endregion

for col in median_fill_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Handling by filling with 0

df['security_deposit'].fillna(0,inplace=True)
df['cleaning_fee'].fillna(0,inplace=True)

# endregion

# region Handling nulls in Text Columns

# By filling with default Phrases
df['neighborhood_overview'].fillna("No neighborhood info", inplace=True)
df['notes'].fillna("No notes", inplace=True)
df['transit'].fillna("No transit info", inplace=True)
df['access'].fillna("No access info", inplace=True)
df['interaction'].fillna("No interaction info", inplace=True)
df['house_rules'].fillna("No house rules", inplace=True)
df['host_about'].fillna("No host info", inplace=True)

# By text Generation using cohere AI
# region Cohere

# import cohere

# co = cohere.Client("juIYpQ3NwQHAyNnlwcLeCzjZGa9tHZUvgj24TL3V")

# def generate_summary(row):
#     if pd.notnull(row['summary']):
#         return row['summary']

#     prompt = f"Write a cozy and inviting one-sentence summary for a {row['bedrooms']}-bedroom place in the {row['smart_location']} with {row['amenities']}."

#     response = co.generate(
#         model='command',
#         prompt=prompt,
#         max_tokens=60,
#         temperature=0.7,
#     )

#     return response.generations[0].text.strip()

# df['summary'] = df.apply(generate_summary, axis=1)

# df.to_csv('GuestSatisfactionPrediction_Copy.csv', index=False)

# co = cohere.Client("h2W3WCDLhT6df3h4ksISWtTIqvRbPfCGAQuf07gA")

#-----------------------------------------------------------------------
# def generate_description(row):
#     if pd.notnull(row['description']):
#         return row['description']

#     prompt = f"""Write a cozy, detailed, and inviting property description for a rental listing.
# It should describe a {row['bedrooms']}-bedroom place located in {row['smart_location']} with the following amenities: {row['amenities']}.
# Highlight features like living spaces, kitchen equipment, outdoor areas, and anything that would make a guest feel at home."""

#     response = co.generate(
#         model='command',
#         prompt=prompt,
#         max_tokens=250,
#         temperature=0.7,
#     )

#     return response.generations[0].text.strip()


# df['description'] = df.apply(generate_description, axis=1)

# df.to_csv('GuestSatisfactionPrediction_Copy.csv', index=False)

#-----------------------------------------------------------------------------------------

# co1 = cohere.Client("yFsJR6NCZmgyJ9KCc1FkjrFt0BOkIXnOnrDDY9R6")  # First API key


# def generate_summary(row):
#     if pd.notnull(row['space']):
#         return row['space']

#     prompt = f"Write a detailed and inviting description of a {row['bedrooms']}-bedroom home in the {row['neighbourhood_cleansed']} neighborhood of {row['city']}, located in {row['smart_location']}. The home features {row['amenities']}. Highlight the house's size, layout, and any special features that make it ideal for families, groups, or individuals."

#     # Try the first API key, then switch to the second if it fails

#     response = co1.generate(
#             model='command',
#             prompt=prompt,
#             max_tokens=150,
#             temperature=0.7,
#         )


#     return response.generations[0].text.strip()

# df['space'] = df.apply(generate_summary, axis=1)
# df.to_csv('GuestSatisfactionPrediction_Copy.csv', index=False)





# endregion

# endregion

# region Flooring Columns

numeric_columns = ['bathrooms', 'bedrooms', 'beds']

for col in numeric_columns:
    df[col] = np.floor(df[col]).astype(int)
    
# endregion

# region Text Processing

# region Processing with NLTK TF-IDF
# Text Cleaning

text_columns = ['summary', 'space', 'description','notes', 'transit', 'access', 'house_rules']
for col in text_columns:
    df[col] = (
        df[col]
        .str.lower() # lowercasing
        .str.replace(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", "", regex=True)  # remove URLs & emails
        .str.replace(r"<.*?>", "", regex=True)  # remove HTML tags
        .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)  # remove punctuation & special chars
        .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
        .str.strip()  # trim leading/trailing spaces
    )


# Handling contradictions

contractions_dict = {
    "isn’t": " is not",
    "don’t": " do not",
    "aren’t": " are not",
    "can’t": " cannot",
    "couldn’t": " could not",
    "didn’t": " did not",
    "’ve": " have",
    "’d": " would",
    "u": " you",
    "’m": " am",
    "’ll":" will",
    "’re": " are",
    "won’t": " will not"

}

def expand_contractions(text):
    for contraction, expansion in contractions_dict.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', expansion, text)
    return text

text_columns = ['summary', 'space', 'description','notes', 'transit', 'access', 'house_rules']
for col in text_columns:
    df[col] = df[col].apply(expand_contractions)
    

# Tokenization

text_columns = ['summary', 'space', 'description','notes', 'transit', 'access', 'house_rules']
for col in text_columns:
    df[col] = df[col].apply(word_tokenize)
    
# Removing Stopwords

stop_words = set(stopwords.words('english'))

def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]

text_columns = ['summary', 'space', 'description','notes', 'transit', 'access', 'house_rules']
for col in text_columns:
    df[col] = df[col].apply(remove_stop_words)

# Lemmatization

lemmatizer = WordNetLemmatizer()

def lemma(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

text_columns = ['summary', 'space', 'description','notes', 'transit', 'access', 'house_rules']

for col in text_columns:
    df[col] = df[col].apply(lemma)
    
# Text Vectorization using TF-IDF

df['summary'] = df['summary'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['space'] = df['space'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['description'] = df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['notes'] = df['notes'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['transit'] = df['transit'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['access'] = df['access'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
df['house_rules'] = df['house_rules'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# summary
summary_tfidf = TfidfVectorizer(max_features=1000)
summary_features = summary_tfidf.fit_transform(df['summary'])
summary_tfidf_df = pd.DataFrame(summary_features.toarray(), columns=[f"summary_{f}" for f in summary_tfidf.get_feature_names_out()])

# space
space_tfidf = TfidfVectorizer(max_features=1000)
space_features = space_tfidf.fit_transform(df['space'])
space_tfidf_df = pd.DataFrame(space_features.toarray(), columns=[f"space_{f}" for f in space_tfidf.get_feature_names_out()])


# description
description_tfidf = TfidfVectorizer(max_features=1000)
description_features = description_tfidf.fit_transform(df['description'])
description_tfidf_df = pd.DataFrame(description_features.toarray(), columns=[f"description_{f}" for f in description_tfidf.get_feature_names_out()])

# notes
notes_tfidf = TfidfVectorizer(max_features=1000)
notes_features = notes_tfidf.fit_transform(df['notes'])
notes_tfidf_df = pd.DataFrame(notes_features.toarray(), columns=[f"notes_{f}" for f in notes_tfidf.get_feature_names_out()])


# transit
transit_tfidf = TfidfVectorizer(max_features=1000)
transit_features = transit_tfidf.fit_transform(df['transit'])
transit_tfidf_df = pd.DataFrame(transit_features.toarray(), columns=[f"transit_{f}" for f in transit_tfidf.get_feature_names_out()])


# access
access_tfidf = TfidfVectorizer(max_features=1000)
access_features = access_tfidf.fit_transform(df['access'])
access_tfidf_df = pd.DataFrame(access_features.toarray(), columns=[f"access_{f}" for f in access_tfidf.get_feature_names_out()])


# house_rules
house_rules_tfidf = TfidfVectorizer(max_features=1000)
house_rules_features = house_rules_tfidf.fit_transform(df['house_rules'])
house_rules_tfidf_df = pd.DataFrame(house_rules_features.toarray(), columns=[f"house_rules_{f}" for f in house_rules_tfidf.get_feature_names_out()])

# Dimensionality Reduction using SVD

svd_summary = TruncatedSVD(n_components=50, random_state=0)
svd_space = TruncatedSVD(n_components=50, random_state=0)
svd_transit = TruncatedSVD(n_components=50, random_state=0)
svd_access = TruncatedSVD(n_components=50, random_state=0)
svd_description = TruncatedSVD(n_components=50, random_state=0)
svd_notes = TruncatedSVD(n_components=50, random_state=0)
svd_house_rules = TruncatedSVD(n_components=50, random_state=0)

summary_svd_df = pd.DataFrame(svd_summary.fit_transform(summary_features), columns=[f"summary_svd_{i}" for i in range(50)])
space_svd_df = pd.DataFrame(svd_space.fit_transform(space_features), columns=[f"space_svd_{i}" for i in range(50)])
transit_svd_df = pd.DataFrame(svd_transit.fit_transform(transit_features), columns=[f"transit_svd_{i}" for i in range(50)])
access_svd_df = pd.DataFrame(svd_access.fit_transform(access_features), columns=[f"access_svd_{i}" for i in range(50)])
description_svd_df = pd.DataFrame(svd_description.fit_transform(description_features), columns=[f"description_svd_{i}" for i in range(50)])
notes_svd_df = pd.DataFrame(svd_notes.fit_transform(notes_features), columns=[f"notes_svd_{i}" for i in range(50)])
house_rules_svd_df = pd.DataFrame(svd_house_rules.fit_transform(house_rules_features), columns=[f"house_rules_svd_{i}" for i in range(50)])

df.reset_index(drop=True, inplace=True)
df = pd.concat([
    df,
    summary_svd_df,
    space_svd_df,
    transit_svd_df,
    access_svd_df,
    description_svd_df,
    notes_svd_df,
    house_rules_svd_df
], axis=1)

print("Shape after nlp processing:", df.shape)

# endregion

# region processing with sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

for column in ['interaction','host_about','neighborhood_overview','name']:
    df[column] = df[column].apply(get_sentiment)

print(df[['interaction','host_about','neighborhood_overview','name']].head(5))


# endregion

# region Standardizing Short text columns

columns_to_standardize = [
    'host_name', 'host_location', 'host_neighbourhood',
    'street', 'neighbourhood', 'neighbourhood_cleansed',
    'city', 'state', 'market', 'smart_location', 'country_code','country','property_type','room_type'
]
for col in columns_to_standardize:
    df[col] = df[col].str.lower().str.strip()


# endregion

# endregion

# region Encoding

# region LabelEncoder

label_encoders = {}

columns_to_encode = [
    'host_neighbourhood', 'street', 'neighbourhood', 'neighbourhood_cleansed',
    'city', 'state', 'market', 'smart_location', 'country_code', 'country',
    'property_type', 'room_type', 'host_name'
]

for col in columns_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le     

# endregion

# region binary_Mapping

columns_to_map = [
    'host_has_profile_pic', 'host_identity_verified', 'is_location_exact',
    'requires_license', 'instant_bookable', 'is_business_travel_ready',
    'require_guest_phone_verification', 'host_is_superhost'
]
mapping_dict = {'t': 1, 'f': 0}
for col in columns_to_map:
    df[col] = df[col].map(mapping_dict)

df['require_guest_profile_picture'] = df['require_guest_profile_picture'].map({'t': 0, 'f': 1})

# endregion

# region Mapping Categorical Data to Numeric Values with Rank (Higher is Better)

guest_satisfaction_mapping={
    'Very High':2,
    'High':1,
    'Average':0,
}
df['guest_satisfaction'] = df['guest_satisfaction'].map(guest_satisfaction_mapping)

response_mapping = {
    'within an hour': 3,
    'within a few hours': 2,
    'within a day': 1,
    'a few days or more': 0
}
df['host_response_time'] = df['host_response_time'].map(response_mapping)

bed_rank = {
    'Airbed': 0,
    'Couch': 1,
    'Futon': 2,
    'Pull-out Sofa': 3,
    'Real Bed': 4
}

df['bed_type'] = df['bed_type'].map(bed_rank)

cancellation_rank = {
    'flexible': 5,
    'moderate': 4,
    'strict': 3,
    'strict_14_with_grace_period': 2,
    'super_strict_30': 1,
    'super_strict_60': 0
}

df['cancellation_policy'] = df['cancellation_policy'].map(cancellation_rank)



# endregion
# endregion
print (df.isnull().sum())
# region Handling Outliers using IQR method

outliers_cols = [
    'host_total_listings_count','host_listings_count',
    "latitude", "longitude",
    "accommodates", "bathrooms", "bedrooms", "beds",
    "guests_included", "minimum_nights",
    "maximum_nights", "number_of_reviews", "number_of_stays"
]

for col in outliers_cols:
  q1 = np.percentile(df[col], 25)
  q3 = np.percentile(df[col], 75)
  norm_range = (q3 - q1) * 1.5
  lower_outliers = df[df[col] < (q1 - norm_range)]
  upper_outliers = df[df[col] > (q3 + norm_range)]
  outliers = len(lower_outliers)+len(upper_outliers)
  print(f"The number of outliers in {col} is : {outliers}")

# region boxplot
chunks = [outliers_cols[i::4] for i in range(4)]

for group in chunks:
    fig, axes = plt.subplots(len(group), 1, figsize=(10, len(group) * 5))  # Make it taller
    if len(group) == 1:
        axes = [axes]

    for ax, col in zip(axes, group):
        sns.boxplot(x=df[col], palette=["#E41D53"], ax=ax)
        ax.set_title(f'Boxplot of {col}', fontsize=14)
        ax.set_xlabel('')  # Remove x-label
        ax.tick_params(axis='x', rotation=45)  # Rotate x-ticks by 45 degrees

    plt.tight_layout(pad=3.0)  # Add padding between plots
    plt.show()

# endregion

for col in outliers_cols:
  q1 = np.percentile(df[col], 25)
  q3 = np.percentile(df[col], 75)
  norm_range = (q3 - q1) * 1.5
  lower_outliers = df[df[col] < (q1 - norm_range)]
  upper_outliers = df[df[col] > (q3 + norm_range)]
  outliers = len(lower_outliers)+len(upper_outliers)
  print(f"The number of outliers in {col} is : {outliers}")
  

# endregion

# endregion

# region Feature_Engineering
# region WebScraping
# Using Beautiful Soup
# import requests
# from bs4 import BeautifulSoup
# import re
# import pandas as pd

# # Let's assume df is already loaded with the 'host_url' column

# def get_host_rating_with_progress(url, idx, total):
#     try:
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
#         }
#         response = requests.get(url, headers=headers, timeout=10)

#         if response.status_code != 200:
#             print(f"[{idx}/{total}] ❌ Failed to access {url}")
#             return None

#         soup = BeautifulSoup(response.content, 'html.parser')

#         rating_span = soup.find('span', string=re.compile(r'(\d+\.\d+)\s+Rating'))

#         if rating_span:
#             rating_text = rating_span.get_text()
#             match = re.search(r'\d+\.\d+', rating_text)
#             if match:
#                 rating_value = float(match.group())
#                 print(f"[{idx}/{total}] ✅ Found rating {rating_value} for {url}")
#                 return rating_value
#             else:
#                 print(f"[{idx}/{total}] ❌ Rating format mismatch for {url}")
#         else:
#             print(f"[{idx}/{total}] ❌ Rating not found for {url}")

#         return None

#     except Exception as e:
#         print(f"[{idx}/{total}] ⚠️ Error scraping {url}: {e}")
#         return None

# # Apply function with index tracking
# total_urls = len(df)
# df['host_rating'] = [
#     get_host_rating_with_progress(url, idx+1, total_urls)
#     for idx, url in enumerate(df['host_url'])
# ]

# output_file = "GuestSatisfactionPrediction.csv"
# df.to_csv(output_file, index=False)

# print(f"✅ Updated data saved to {output_file}")

# Using Selenium

# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.common.by import By
# import pandas as pd
# import time

# # Setup headless Chrome
# options = Options()
# options.add_argument("--headless")
# options.add_argument("--no-sandbox")
# options.add_argument("--disable-dev-shm-usage")

# driver = webdriver.Chrome(options=options)

# def has_guest_favorite(url, idx, total):
#     try:
#         driver.get(url)
#         time.sleep(3)


#         try:
#             driver.find_element(By.CSS_SELECTOR, "[data-section-id='TITLE_DEFAULT']")
#             listing_page = True
#         except:
#             listing_page = False

#         if not listing_page:
#             print(f"[{idx}/{total}] ⚠️ Not a listing page (likely redirected to homepage) for {url}")
#             return 0

#         # Now check for Guest favorite
#         try:
#             guest_fav = driver.find_element(By.XPATH, "//div[contains(text(),'Guest favorite')]")
#             print(f"[{idx}/{total}] ✅ 'Guest favorite' FOUND for {url}")
#             return 1
#         except:
#             print(f"[{idx}/{total}] ❌ 'Guest favorite' NOT FOUND for {url}")
#             return 0

#     except Exception as e:
#         print(f"[{idx}/{total}] ⚠️ Error scraping {url}: {e}")
#         return None

# # Apply function with index tracking
# total_urls = len(df)
# df['guest_favorite'] = [
#     has_guest_favorite(url, idx+1, total_urls)
#     for idx, url in enumerate(df['listing_url'])
# ]

# driver.quit()

# output_file = "GuestSatisfactionPrediction.csv"
# df.to_csv(output_file, index=False)

# print(f"✅ Updated data saved to {output_file}")





# endregion
# region Dividing amenities into categories

all_amenities = set()

for amenities in df['amenities']:
    amenities = amenities.strip()
    if amenities.startswith("{") and amenities.endswith("}"):
        amenities = amenities[1:-1] 

    items = [item.strip().strip('"').strip("'") for item in amenities.split(",")]
    all_amenities.update(items)

for amenity in sorted(all_amenities):
    print(amenity)
    
amenities_dict = {
    "Essentials": [
        "Essentials", "Bath towel", "Bathroom essentials", "Bed linens", "Bedroom comforts",
        "Body soap", "Cooking basics", "Dishes and silverware", "Hangers", "Heating",
        "Hot water", "Internet", "Shampoo", "Toilet paper", "Wifi", "TV", "Cleaning before checkout",
        "Ethernet connection", "Hair dryer", "Hot water kettle", "toilet"
    ],
    "Safety": [
        "Carbon monoxide detector", "Fire extinguisher", "First aid kit", "Safety card",
        "Smoke detector", "Window guards", "Buzzer/wireless intercom",
        "Lock on bedroom door", "Doorman", "Smart lock", "Keypad", "Fireplace guards"
    ],
    "Luxury": [
        "Air purifier", "Alfresco bathtub", "En suite bathroom", "Espresso machine", "Firm mattress",
        "Heated floors", "Heated towel rack", "Hot tub", "Jetted tub", "Memory foam mattress",
        "Pillow-top mattress", "Private hot tub", "Private pool", "Rain shower", "Sauna",
        "Soaking tub", "Sound system", "Stand alone steam shower", "Sun loungers", "Wine cooler",
        "Building staff", "Day bed", "Host greets you", "Indoor fireplace", "Luggage dropoff allowed",
        "Private bathroom", "Private entrance", "Private living room", "Room-darkening shades",
        "Suitable for events", "Ski-in/Ski-out", "Smoking allowed"
    ],
    "Accessibility": [
        "24-hour check-in", "Accessible-height bed", "Accessible-height toilet", "Disabled parking spot",
        "Electric profiling bed", "Elevator", "Extra space around bed", "Flat path to guest entrance",
        "Ground floor access", "Handheld shower head", "No stairs or steps to enter",
        "Pool with pool hoist", "Roll-in shower", "Shower chair", "Single level home",
        "Well-lit path to entrance", "Wheelchair accessible", "Wide clearance to shower",
        "Wide doorway to guest bathroom", "Wide entrance", "Wide entrance for guests",
        "Wide entryway", "Wide hallways", "Fixed grab bars for shower", "Fixed grab bars for toilet",
        "Bathtub with bath chair"
    ],
    "Outdoor": [
        "BBQ grill", "Balcony", "Beach essentials", "Beach view", "Beachfront",
        "Free parking on premises", "Free street parking", "Garden or backyard", "Hammock",
        "Lake access", "Mountain view", "Outdoor kitchen", "Outdoor parking",
        "Outdoor seating", "Patio or balcony", "Terrace", "Waterfront", "Tennis court",
        "Pool", "Pool toys", "Fire pit"
    ],
    "Child & Family-Friendly": [
        "Baby bath", "Baby monitor", "Babysitter recommendations", "Changing table",
        "Children's books and toys", "Children's dinnerware", "Crib", "Family/kid friendly",
        "High chair", "Outlet covers", "Pack 'n Play/travel crib", "Stair gates",
        "Table corner guards", "Other pet(s)", "Pets allowed", "Cat(s)", "Dog(s)", "Pets live on this property"
    ],
    "Entertainment": [
        "Amazon Echo", "Cable TV", "DVD player", "Game console", "HBO GO",
        "Netflix", "Projector and screen", "Smart TV"
    ],
    "Home Appliances": [
        "Air conditioning", "Ceiling fan", "Central air conditioning", "Coffee maker",
        "Convection oven", "Dishwasher", "Dryer", "EV charger", "Exercise equipment",
        "Fax machine", "Full kitchen", "Gas oven", "Gym", "High-resolution computer monitor",
        "Kitchen", "Kitchenette", "Laptop friendly workspace", "Lockbox", "Long term stays allowed",
        "Microwave", "Mini fridge", "Murphy bed", "Oven", "Paid parking off premises",
        "Paid parking on premises", "Printer", "Refrigerator", "Stove", "Washer",
        "Warming drawer", "Pocket wifi", "Shared gym", "Shared hot tub", "Shared pool",
        "Self check-in", "Extra pillows and blankets", "Formal dining area", "Standing valet",
        "Iron", "Double oven", "Heat lamps", "Breakfast", "Breakfast table", "Bidet"
    ]
}

amenity_to_category = {}
for category, amenities in amenities_dict.items():
    for amenity in amenities:
        amenity_to_category[amenity] = category

all_categories = list(amenities_dict.keys())

def categorize_amenities(amenities_str):
    if pd.isna(amenities_str):
        amenities = []
    else:
        amenities_str = amenities_str.strip()
        if amenities_str.startswith("{") and amenities_str.endswith("}"):
            amenities_str = amenities_str[1:-1] 
        amenities = [item.strip().strip('"').strip("'") for item in amenities_str.split(",")]

    counts = defaultdict(int)
    for amenity in amenities:
        if amenity:  
            category = amenity_to_category.get(amenity)
            if category:
                counts[category] += 1

    full_counts = {cat: counts.get(cat, 0) for cat in all_categories}
    return pd.Series(full_counts)

category_counts_df = df["amenities"].apply(categorize_amenities)

df = pd.concat([df, category_counts_df], axis=1)

print(df[['amenities','Essentials','Safety','Luxury','Accessibility','Outdoor','Child & Family-Friendly','Entertainment','Home Appliances']].head(5))

print(df.shape)


# endregion
# region Extracting New features from date columns
# Extracting number of active years from host since column

df['host_since'] = pd.to_datetime(df['host_since'], format='%m/%d/%Y')

today = pd.to_datetime(datetime.today())
df['years_active'] = (today - df['host_since']).dt.days / 365

df['years_active'] = df['years_active'].round(1)

print(df[['years_active','host_since']].head())

# Extracting review frequency per day from last_review,first_review and number_of_reviews columns

df['first_review'] = pd.to_datetime(df['first_review'], format='%m/%d/%Y')
df['last_review'] = pd.to_datetime(df['last_review'], format='%m/%d/%Y')

df['reviews_per_day'] = df.apply(
    lambda row: row['number_of_reviews'] / ((row['last_review'] - row['first_review']).days)
    if (row['last_review'] - row['first_review']).days != 0 else 0,
    axis=1
)

df['reviews_per_day'] = df['reviews_per_day'].round(4)

print(df[['reviews_per_day', 'first_review', 'last_review', 'number_of_reviews']].head())

# Dividing host location column into host city , host state and host country

print(df['host_location'].unique())

def split_host_location(location):

    if location in ['us', 'ca', 'at', 'gb', 'mx', 'it', 'de', 'canada', 'china', 'mexico', 'southern california']:
        return (np.nan, np.nan, location)

    if len(location.split(',')) < 1 or len(location.split(',')) > 4:
        return (np.nan, np.nan, np.nan)

    parts = [p.strip() for p in location.split(',')]

    if len(parts) == 3:
        city, state, country = parts
    elif len(parts) == 2:
        city = np.nan
        state, country = parts
    elif len(parts) == 1:
        city = np.nan
        state = np.nan
        country = parts[0]
    else:
        city = np.nan
        state = np.nan
        country = np.nan

    return (city, state, country)

df[['host_city', 'host_state', 'host_country']] = df['host_location'].apply(split_host_location).apply(pd.Series)

df[['host_location', 'host_city', 'host_state', 'host_country']].head()


le = LabelEncoder()
columns_to_encode = ['host_city', 'host_state', 'host_country']

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

print(df[['host_location', 'host_city', 'host_state', 'host_country']].head())

df['total_cost'] = (
    df['price_per_stay'] +
    df['cleaning_fee'] +
    df['security_deposit']
)
print(df[['price_per_stay', 'cleaning_fee', 'security_deposit','total_cost']].head())


# endregion
# endregion

# region Feature Selection
# Dropping un-necessary columns

# Dropping id and host_id as they are unique indetifiers

is_id_unique = df['id'].is_unique
print(f"Is 'id' unique? {is_id_unique}")
print(f"Number of rows: {len(df)}")
print(f"Unique 'id' values: {df['id'].nunique()}")

# Dropping thumbnail_url and host_acceptance_rate as these columns are entirely null

total_rows = len(df)
thumbnail_url_null_count = df['thumbnail_url'].isnull().sum()
host_acceptance_rate_null_count = df['host_acceptance_rate'].isnull().sum()
print(f"Total rows in data: {total_rows}")
print(f"Null values in 'thumbnail_url': {thumbnail_url_null_count}")
print(f"Null values in 'host_acceptance_rate': {host_acceptance_rate_null_count}")

# Dropping square_feet as it contains a majority of null values

null_percentage_square_feet = df['square_feet'].isnull().mean() * 100
print(f"Percentage of null values in 'square_feet': {null_percentage_square_feet:.2f}%")

# Dropping host_listings_count as the same as host_total_listings_count

identical_columns = df['host_listings_count'].equals(df['host_total_listings_count'])
if identical_columns:
    print("The columns 'host_listings_count' and 'host_total_listings_count' are identical")
else:
    print("The columns 'host_listings_count' and 'host_total_listings_count' are not identical.")
    
df = df.drop(['id','listing_url','host_url','thumbnail_url','host_acceptance_rate','square_feet','summary','space','transit','access','host_since','host_location',
              'host_listings_count','amenities','first_review','last_review','description','notes','house_rules'], axis=1)

# region Feature Selection on Discrete Features

discrete_features = [
    'host_name','host_response_time',
    'host_is_superhost',
    'host_neighbourhood', 
    'host_has_profile_pic', 'host_identity_verified', 'street', 'neighbourhood',
    'neighbourhood_cleansed', 'city', 'state', 'market', 'smart_location',
    'country_code', 'country', 'is_location_exact', 'property_type', 'room_type',
    'bed_type', 'requires_license',
    'instant_bookable', 'is_business_travel_ready', 'cancellation_policy',
    'require_guest_profile_picture', 'require_guest_phone_verification', 'guest_favorite'
]

X_discrete = df[discrete_features]
y_discrete = df['guest_satisfaction']

# Chi Squared

chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X_discrete, y_discrete)

chi2_scores = pd.DataFrame({
    'Feature': X_discrete.columns,
    'Chi2 Score': chi2_selector.scores_,
    'P-Value': chi2_selector.pvalues_
}).sort_values(by='Chi2 Score', ascending=False)

chi_features = chi2_scores[chi2_scores['P-Value'] < 0.05]

sorted_features =chi_features.sort_values(by='Chi2 Score', ascending=False).reset_index(drop=True)

print("Top Chi² Features:")
print(sorted_features)

chi2_scores_sorted = chi2_scores.sort_values(by='Chi2 Score', ascending=False).reset_index(drop=True)
features = chi2_scores_sorted['Feature']
scores = chi2_scores_sorted['Chi2 Score']
colors = ['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0']
color_cycle = [colors[i % len(colors)] for i in range(len(features))]
fig = go.Figure(go.Bar(
    x=scores,
    y=features,
    orientation='h',
    marker=dict(color=color_cycle)
))
fig.update_layout(
    title='Chi² Feature Importance (All Discrete Features)',
    xaxis_title='Chi² Score',
    yaxis=dict(autorange='reversed'),
    template='plotly_white',
    height=30 * len(features)
)

fig.show()

# Mutual Information

mi_scores = mutual_info_classif(X_discrete, y_discrete,random_state=0)
mi_scores_df = pd.DataFrame({
    'Feature': X_discrete.columns,
    'Mutual Information': mi_scores
}).sort_values(by='Mutual Information', ascending=False)

threshold = 0.01
mutual_info_features = mi_scores_df[mi_scores_df['Mutual Information'] > threshold]
print(mutual_info_features)

# region Plot
mi_scores_df_sorted = mi_scores_df.sort_values(by='Mutual Information', ascending=False).reset_index(drop=True)

colors = ['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0']
color_cycle = [colors[i % len(colors)] for i in range(len(mi_scores_df_sorted))]

fig = go.Figure(go.Bar(
    x=mi_scores_df_sorted['Mutual Information'],
    y=mi_scores_df_sorted['Feature'],
    orientation='h',
    marker=dict(color=color_cycle)
))

fig.update_layout(
    title='Feature Importance via Mutual Information',
    xaxis_title='Mutual Information Score',
    yaxis=dict(autorange='reversed'),  
    template='plotly_white',
    height=30 * len(mi_scores_df_sorted) 
)

# endregion

# endregion

# region Feature Selection on Continous Features

cont_features = []
for col in df.columns:
    if col not in discrete_features and col != 'guest_satisfaction':
        cont_features.append(col)
        
X_cont = df[cont_features]
y = df['guest_satisfaction']

# Anova

f_values, p_values = f_classif(X_cont, y)
anova_results = pd.DataFrame({
    'Feature': cont_features,
    'F-Value': f_values,
    'P-Value': p_values
})
anova_results_sorted = anova_results.sort_values(by='F-Value', ascending=False)
f_threshold = 50
anova_features = anova_results_sorted[anova_results_sorted['F-Value'] >= f_threshold]
print(anova_features)

# region AnovaPlot 

anova_results_sorted = anova_results.sort_values(by='F-Value', ascending=False)

colors = ['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0']
color_cycle = [colors[i % len(colors)] for i in range(len(anova_results_sorted))]

fig = go.Figure(go.Bar(
    x=anova_results_sorted['F-Value'],
    y=anova_results_sorted['Feature'],
    orientation='h',
    marker=dict(color=color_cycle)
))

fig.update_layout(
    title='Feature Importance via ANOVA F-test',
    xaxis_title='F-Value',
    yaxis=dict(autorange='reversed'),  
    height=30 * len(anova_results_sorted),  
    template='plotly_white'
)

fig.show()

# endregion

# endregion

# endregion

# region Using Chi square and Anova for final features

selected_features = set(chi_features['Feature']).union(set(anova_features['Feature']))

# Print the combined selected features
selected_features=list(selected_features)
print("Selected Features from both methods:")
print(selected_features)

# endregion

# region Model Training and Selection

# Splitting the data into features and target

x=df[selected_features]
y= df['guest_satisfaction']



# Splitting the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature Scaling using Robust Scaler


scaler = RobustScaler()
# scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handling Imbalanced Classes

print(df['guest_satisfaction'].value_counts())

# region piplot 
plt.figure(figsize=(7, 7))
df['guest_satisfaction'].value_counts().plot.pie(
    autopct='%1.1f%%', 
    colors=['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], 
    startangle=90
)
plt.axis('equal')
plt.title('Guest Satisfaction Distribution')
plt.show()
# endregion

smote = SMOTE(random_state=0)
X_train, y_train = smote.fit_resample(X_train, y_train)

# endregion

# region Linear Models

# region Logistic Regression

logistic_reg = LogisticRegression(random_state=0)


cv_scores = cross_val_score(logistic_reg, x, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-Validation Accuracy Scores: {cv_scores}")
print(f"Average Cross-Validation Accuracy: {cv_scores.mean():.4f}")

start_train = time.time()
logistic_reg.fit(X_train, y_train)
end_train = time.time()
train_time_log = end_train - start_train
print(f"Training Time: {train_time_log:.4f} seconds")

train_score = logistic_reg.score(X_train, y_train)
print(f"Training Accuracy Score: {train_score:.4f}")

start_test = time.time()
y_pred_log = logistic_reg.predict(X_test)
end_test = time.time()
test_time_log = end_test - start_test
print(f"Test Time: {test_time_log:.4f} seconds")

test_score_log = accuracy_score(y_test, y_pred_log)
print(f"Test Accuracy Score: {test_score_log:.4f}")
print("Test Set Classification Report:")
print(classification_report(y_test, y_pred_log))

# region CMLR

cm = confusion_matrix(y_test, y_pred_log)
custom_cmap = sns.color_palette(['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], as_cmap=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


# endregion



# endregion

# region SVC

param_grid_svc = {
    'C': [1, 10, 100],                
    'kernel': ['rbf', 'poly', 'sigmoid']  
}

svc = SVC(class_weight='balanced')         

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)  

grid_search_svc = GridSearchCV(estimator=svc, param_grid=param_grid_svc, cv=cv, scoring='accuracy', n_jobs=-1)

start_train = time.time()
grid_search_svc.fit(X_train, y_train)
end_train = time.time()
train_time_svc = end_train - start_train
print(f"Total Training Time (SVC): {train_time_svc:.4f} seconds")

best_params_svc = grid_search_svc.best_params_
print(f"Best parameters (SVC): {best_params_svc}")
best_score_svc = grid_search_svc.best_score_
print(f"Best CV score (Accuracy) (SVC): {best_score_svc:.4f}")

best_svc_model = grid_search_svc.best_estimator_

train_accuracy_svc = best_svc_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_svc:.4f}")

start_test = time.time()
y_pred_svc = best_svc_model.predict(X_test)
end_test = time.time()
test_time_svc = end_test - start_test
print(f"Total Test Time (SVC): {test_time_svc:.4f} seconds")

print("Test Accuracy Score (SVC):")
print(accuracy_score(y_test, y_pred_svc))
print("Classification Report (SVC):")
print(classification_report(y_test, y_pred_svc))

# region CMSVC

cm_svc = confusion_matrix(y_test, y_pred_svc)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svc, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (SVC)')
plt.tight_layout()
plt.show()

# endregion

# endregion


# endregion

# region Ensemble models

# region Bagging models

# region RandomForest




param_grid_rf = {
    'n_estimators': [50, 100,200],  
    'max_depth': [None, 10],  
    'min_samples_split': [2, 5,10], 
    'min_samples_leaf': [1, 2]  
}

rf = RandomForestClassifier(random_state=0)

grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='accuracy', n_jobs=-1)

start_train = time.time()
grid_search_rf.fit(X_train, y_train)
end_train = time.time()
train_time_rf = end_train - start_train
print(f"Total Training Time (RandomForest): {train_time_rf:.4f} seconds")

best_params_rf = grid_search_rf.best_params_
print(f"Best parameters (RandomForest): {best_params_rf}")
best_score_rf = grid_search_rf.best_score_
print(f"Best CV score (Accuracy) (RandomForest): {best_score_rf:.4f}")

best_rf_model = grid_search_rf.best_estimator_

train_accuracy_rf = best_rf_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_rf:.4f}")

start_test = time.time()
y_pred_rf = best_rf_model.predict(X_test)
end_test = time.time()
test_time_rf = end_test - start_test
print(f"Total Test Time (RandomForest): {test_time_rf:.4f} seconds")

print("Test Accuracy Score (RandomForest):")
print(accuracy_score(y_test, y_pred_rf))
print("Classification Report (RandomForest):")
print(classification_report(y_test, y_pred_rf))

# region CMRF

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest)')
plt.tight_layout()
plt.show()

# endregion


# endregion

# region DesicionTree

param_grid_dt = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt', 'log2']
}

dt = DecisionTreeClassifier(random_state=0)

grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)

start_train = time.time()
grid_search_dt.fit(X_train, y_train)
end_train = time.time()
train_time_dt = end_train - start_train
print(f"Total Training Time (DecisionTree): {train_time_dt:.4f} seconds")

best_params_dt = grid_search_dt.best_params_
print(f"Best parameters (DecisionTree): {best_params_dt}")
best_score_dt = grid_search_dt.best_score_
print(f"Best CV score (Accuracy) (DecisionTree): {best_score_dt:.4f}")

best_dt_model = grid_search_dt.best_estimator_

train_accuracy_dt = best_dt_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_dt:.4f}")

start_test = time.time()
y_pred_dt = best_dt_model.predict(X_test)
end_test = time.time()
test_time_dt = end_test - start_test
print(f"Total Test Time (DecisionTree): {test_time_dt:.4f} seconds")

print("Test Accuracy Score (DecisionTree):")
print(accuracy_score(y_test, y_pred_dt))
print("Classification Report (DecisionTree):")
print(classification_report(y_test, y_pred_dt))

# region CMDT

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_dt, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Decision Tree)')
plt.tight_layout()
plt.show()

# endregion






# endregion

# endregion

# region Boosting models

# region XGBOOST

xgb_model = XGBClassifier(
    n_estimators=75,
    max_depth=6,
    learning_rate=0.10063569413953903,
    subsample=0.7773356344475959,
    colsample_bytree=0.9268996994266611,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=0,
    verbosity=0
)

cv_scores_xgb = cross_val_score(xgb_model, x, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-Validation Accuracy Scores: {cv_scores_xgb}")
print(f"Average Cross-Validation Accuracy: {cv_scores_xgb.mean():.4f}")

start_train = time.time()
xgb_model.fit(X_train, y_train)
end_train = time.time()
train_time_xgb = end_train - start_train
print(f"Total Training Time (XGBoost): {train_time_xgb:.4f} seconds")

train_accuracy_xgb = xgb_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_xgb:.4f}")

start_test = time.time()
y_pred_xgb = xgb_model.predict(X_test)
end_test = time.time()
test_time_xgb = end_test - start_test
print(f"Total Test Time (XGBoost): {test_time_xgb:.4f} seconds")

print("Test Accuracy Score (XGBoost):")
print(accuracy_score(y_test, y_pred_xgb))
print("Test Set Classification Report (XGBoost):")
print(classification_report(y_test, y_pred_xgb))

# region CMXGB

cmxg = confusion_matrix(y_test, y_pred_xgb)
custom_cmap = sns.color_palette(['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], as_cmap=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cmxg, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (XGBoost)')
plt.tight_layout()
plt.show()

# endregion

# endregion

# region CatBOOST

cat_model = CatBoostClassifier(
    depth=5,
    iterations=156,
    learning_rate=0.1594827161532743,
    l2_leaf_reg=6.544844148450393,
    bagging_temperature=0.5394298973201219,
    random_strength=1.7448624019847634,
    border_count=208,
    random_state=0,
    verbose=0
)

cv_scores_cat = cross_val_score(cat_model, x, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-Validation Accuracy Scores: {cv_scores_cat}")
print(f"Average Cross-Validation Accuracy: {cv_scores_cat.mean():.4f}")

start_train = time.time()
cat_model.fit(X_train, y_train)
end_train = time.time()
train_time_cat = end_train - start_train
print(f"Total Training Time (CatBoost): {train_time_cat:.4f} seconds")

train_accuracy_cat = cat_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_cat:.4f}")

start_test = time.time()
y_pred_cat = cat_model.predict(X_test)
end_test = time.time()
test_time_cat = end_test - start_test
print(f"Total Test Time (CatBoost): {test_time_cat:.4f} seconds")

print("Test Accuracy Score (CatBoost):")
print(accuracy_score(y_test, y_pred_cat))
print("Classification Report (CatBoost):")
print(classification_report(y_test, y_pred_cat))

# region CMCB

cmcb = confusion_matrix(y_test, y_pred_cat)
custom_cmap = sns.color_palette(['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], as_cmap=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cmcb, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (CatBoost)')
plt.tight_layout()
plt.show()

# endregion 


# endregion

# region LightGBM

light_model = LGBMClassifier(
    n_estimators=172,
    num_leaves=103,
    learning_rate=0.04485202298566927,
    max_depth=5,
    min_child_samples=43,
    subsample=0.6353937353165494,
    random_state=0,
    colsample_bytree=0.8161981454506166
)

cv_scores_light = cross_val_score(light_model, x, y, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Cross-Validation Accuracy Scores: {cv_scores_light}")
print(f"Average Cross-Validation Accuracy: {cv_scores_light.mean():.4f}")

start_train = time.time()
light_model.fit(X_train, y_train)
end_train = time.time()
train_time_light = end_train - start_train
print(f"Total Training Time (LightGBM): {train_time_light:.4f} seconds")

train_accuracy_light = light_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_light:.4f}")

start_test = time.time()
y_pred_light = light_model.predict(X_test)
end_test = time.time()
test_time_light = end_test - start_test
print(f"Total Test Time (LightGBM): {test_time_light:.4f} seconds")

print("Test Accuracy Score (LightGBM):")
print(accuracy_score(y_test, y_pred_light))
print("Classification Report (LightGBM):")
print(classification_report(y_test, y_pred_light))

# region CMLGB

cmlgb = confusion_matrix(y_test,  y_pred_light)
custom_cmap = sns.color_palette(['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], as_cmap=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cmlgb, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (LightGBM)')
plt.tight_layout()
plt.show()

# endregion


# endregion

# endregion

# region Stacking Model


lgb_model = LGBMClassifier(
    n_estimators=1534,
    learning_rate=0.010382774494101006,
    max_depth=3,
    reg_lambda=35,
    subsample=0.8148197144286347,
    subsample_freq=5,
    random_state=0
)

cat_model = CatBoostClassifier(
    depth=4,
    iterations=2197,
    learning_rate=0.009439768196125256,
    l2_leaf_reg=58,
    bagging_temperature=1.0063058598329384,
    random_state=0,
    verbose=0
)

meta_model = XGBClassifier(
    n_estimators=160,
    max_depth=3,
    learning_rate=0.09290850356426546,
    subsample=0.9207555916472224,
    colsample_bytree=0.9449957597418881,
    random_state=0,
    verbosity=0
)

stacked_model3 = StackingClassifier(
    estimators=[('lightgbm', lgb_model), ('catboost', cat_model)],
    final_estimator=meta_model,
    cv=3,
    n_jobs=-1,
    passthrough=True  
)

start_train_stack = time.time()
stacked_model3.fit(X_train, y_train)
end_train_stack = time.time()
train_time_stack = end_train_stack - start_train_stack
print(f"Total Training Time (Stacking): {train_time_stack:.4f} seconds")

train_accuracy_stack = stacked_model3.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_stack:.4f}")

start_test_stack = time.time()
y_pred_stack3 = stacked_model3.predict(X_test)
end_test_stack = time.time()
test_time_stack = end_test_stack - start_test_stack
print(f"Total Test Time (Stacking): {test_time_stack:.4f} seconds")


print("Test Accuracy Score (Stacking):")
print(accuracy_score(y_test, y_pred_stack3))
print("Classification Report (Stacking):")
print(classification_report(y_test, y_pred_stack3))


# region CMSM

cms = confusion_matrix(y_test,  y_pred_stack3)
custom_cmap = sns.color_palette(['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], as_cmap=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cms, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Stacking model)')
plt.tight_layout()
plt.show()

# endregion






# endregion

# region Voting Model

lgb_model = LGBMClassifier(
    n_estimators=172,
    num_leaves=103,
    learning_rate=0.04485202298566927,
    max_depth=5,
    min_child_samples=43,
    subsample=0.6353937353165494,
    random_state=0,
    colsample_bytree=0.8161981454506166
)

cat_model = CatBoostClassifier(
    depth=5,
    iterations=156,
    learning_rate=0.1594827161532743,
    l2_leaf_reg=6.544844148450393,
    bagging_temperature=0.5394298973201219,
    random_strength=1.7448624019847634,
    border_count=208,
    random_state=0,
    verbose=0
)

xgb_model = XGBClassifier(
    n_estimators=75,
    max_depth=6,
    learning_rate=0.10063569413953903,
    subsample=0.7773356344475959,
    colsample_bytree=0.9268996994266611,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=0,
    verbosity=0
)

voting_model = VotingClassifier(
    estimators=[('lightgbm', lgb_model), ('catboost', cat_model), ('xgboost', xgb_model)],
    voting='soft', 
    n_jobs=-1,

)

start_train_vote = time.time()
voting_model.fit(X_train, y_train)
end_train_vote = time.time()
train_time_vote = end_train_vote - start_train_vote
print(f"Total Training Time (Voting): {train_time_vote:.4f} seconds")

train_accuracy_vote = voting_model.score(X_train, y_train)
print(f"Training Accuracy Score: {train_accuracy_vote:.4f}")

start_test_vote = time.time()
y_pred_vote = voting_model.predict(X_test)
end_test_vote = time.time()
test_time_vote = end_test_vote - start_test_vote
print(f"Total Test Time (Voting): {test_time_vote:.4f} seconds")

print("Test Accuracy Score (Voting):")
print(accuracy_score(y_test, y_pred_vote))

print("Classification Report (Voting):")
print(classification_report(y_test, y_pred_vote))

# region CMV

cmv = confusion_matrix(y_test,  y_pred_vote)
custom_cmap = sns.color_palette(['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'], as_cmap=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cmv, annot=True, fmt='d', cmap=custom_cmap, cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Voting)')
plt.tight_layout()
plt.show()

# endregion

# endregion

# endregion

# region Model Accuracy Comparison

results = {
    'Logistic Regression': accuracy_score(y_test, y_pred_log),
    'SVC': accuracy_score(y_test, y_pred_svc),
    'Random Forest': accuracy_score(y_test, y_pred_rf),
    'Decision Tree': accuracy_score(y_test, y_pred_dt),
    'XGBoost': accuracy_score(y_test, y_pred_xgb),
    'CatBoost': accuracy_score(y_test, y_pred_cat),
    'LightGBM': accuracy_score(y_test, y_pred_light),
    'Stacking': accuracy_score(y_test, y_pred_stack3),
    'Voting': accuracy_score(y_test, y_pred_vote)
}

# Accuracy scores

accuracy_scores = {model: round(results[model], 4) for model in results}
accuracy_table = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy Score'])

print("Accuracy Scores Table: \n ")

print(accuracy_table)

print('\n')

# region Plot 

plt.figure(figsize=(10, 6))
plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='#40E0D0')
plt.title('Model Accuracy Scores', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# endregion

# endregion

# region Model Time comparison

# Training time
train_times = {
    'Logistic Regression': train_time_log,
    'SVC': train_time_svc,
    'Random Forest': train_time_rf,
    'Decision Tree': train_time_dt,
    'XGBoost': train_time_xgb,
    'CatBoost': train_time_cat,
    'LightGBM': train_time_light,
    'Stacking': train_time_stack,
    'Voting': train_time_vote
}

train_times = {model:train_times[model] for model in train_times}
train_times_table = pd.DataFrame(list(train_times.items()), columns=['Model', 'Training Time (seconds)'])

print("Training Times Table: \n ")

print(train_times_table)
print('\n')

# region plot time

plt.figure(figsize=(10, 6))
plt.bar(train_times.keys(), train_times.values(), color='#EE2D6B', label='Training Time')
plt.title('Model Training Times', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Training Time (seconds)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# endregion

# testing times

test_times = {
    'Logistic Regression': test_time_log,
    'SVC': test_time_svc,
    'Random Forest': test_time_rf,
    'Decision Tree': test_time_dt,
    'XGBoost': test_time_xgb,
    'CatBoost': test_time_cat,
    'LightGBM': test_time_light,
    'Stacking': test_time_stack,
    'Voting': test_time_vote
}

test_times = {model: test_times[model] for model in test_times}
test_times_table = pd.DataFrame(list(test_times.items()), columns=['Model', 'Testing Time (seconds)'])
print("Testing Times Table: \n")
print(test_times_table)
print('\n')

# region plottrain
plt.figure(figsize=(10, 6))
plt.bar(test_times.keys(), test_times.values(), color='#F76EA0', label='Testing Time')
plt.title('Model Testing Times', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Testing Time (seconds)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# endregion


# endregion
