#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import math
import os
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore', category=UserWarning, module='lightgbm')
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
pd.set_option('display.max_rows', None)
#Data Exploration
df = pd.read_csv("GuestSatisfactionPrediction.csv")
print("Data Shape: ")
print(df.shape)
print("Data Info: ")
print(df.info())
print("Data Columns: ")
print(df.columns)
print("Data Head: ")
print(df.head())
print("Data Tail: ")
print(df.tail())
print("Data Sample: ")
print(df.sample())
print("Data Description (Excluding object): ")
print(df.describe(exclude='object'))
print("Data Description (Excluding numerical values): ")
print(df.describe(exclude='number'))
#-----------------------------------------------------------------------------------------------------------------------------------
#Data Preprocessing
#------------------------------------------------------------------------------------------------------------------------------------

# region Checking duplicates 

  #1- Checking duplicates on WHOLE data
  
print("whole data duplicates: ",df.duplicated().sum())

 #2-Checking duplicates on ID column
 
print("ID duplicates : " ,df.duplicated(subset='id').sum())

# endregion

# region Fixing numerical columns

df['host_response_rate'] = df['host_response_rate'].str.replace('%', '').astype(float)

currency_columns = ['nightly_price', 'price_per_stay', 'extra_people', 'security_deposit', 'cleaning_fee']

for col in currency_columns:
    df[col] = df[col].str.replace(',', '').str.replace('$', '').astype(float)
    
df['zipcode']=df['zipcode'].str.replace("-", "")

df['zipcode']=df['zipcode'].astype(float)

# endregion

# region Handling Nulls in numerical columns

# 1- Handling nulls using mode 

mode_fill_columns = ['market', 'host_neighbourhood', 'state', 'neighbourhood', 'host_location','host_response_time','zipcode','host_since','host_is_superhost','host_has_profile_pic','host_identity_verified','host_name']

for col in mode_fill_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
    
# 2- Handling nulls using median
#NOTE: We chose to fill missing values using the median instead of the mean because the distribution of several columns showed skewness.

median_fill_columns = [
    'bathrooms', 'bedrooms', 'beds',
    'host_rating', 'host_listings_count', 'host_total_listings_count',
    'host_response_rate'
]


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

#  Filling the nulls with median

for col in median_fill_columns:
    df[col].fillna(df[col].median(), inplace=True)

#  3-Handling nulls by filling it with zero

#NOTE:For `security_deposit` and `cleaning_fee`, filling with `0` made sense logically, as the absence of a value likely indicates no fee was required.  
#while acknowledging that the actual data is unavailable.


df['security_deposit'].fillna(0,inplace=True)
df['cleaning_fee'].fillna(0,inplace=True)
# endregion

#  region Handling nulls in text columns

# region 1) Filling with Default Phrases

#NOTE: For other text fields such as `neighborhood_overview`, `notes`, `transit`, `access`, `interaction`, `house_rules`, and `host_about`, 
#we filled missing values with default placeholders like `"No neighborhood info"` or `"No house rules"`.  
#These placeholders indicate missing data without introducing misleading or artificial content, while still maintaining data consistency and structure.

df['neighborhood_overview'].fillna("No neighborhood info", inplace=True)
df['notes'].fillna("No notes", inplace=True)
df['transit'].fillna("No transit info", inplace=True)
df['access'].fillna("No access info", inplace=True)
df['interaction'].fillna("No interaction info", inplace=True)
df['house_rules'].fillna("No house rules", inplace=True)
df['host_about'].fillna("No host info", inplace=True)
# endregion 
# region 2) Text Generation using Cohere API

#NOTE:For key descriptive fields like `space`, `description`, and `summary`, 
#we used Cohere's language generation capabilities to generate meaningful replacements.
#This helped preserve the quality and completeness of listings where descriptive content is important.

#------SUMMARY-------#
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

#------DESCRIPTION-------#
# import cohere
# co = cohere.Client("h2W3WCDLhT6df3h4ksISWtTIqvRbPfCGAQuf07gA")

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


#------SPACE-------#

# co1 = cohere.Client("yFsJR6NCZmgyJ9KCc1FkjrFt0BOkIXnOnrDDY9R6") 


# def generate_summary(row):
#     if pd.notnull(row['space']):
#         return row['space']

#     prompt = f"Write a detailed and inviting description of a {row['bedrooms']}-bedroom home in the {row['neighbourhood_cleansed']} neighborhood of {row['city']}, located in {row['smart_location']}. The home features {row['amenities']}. Highlight the house's size, layout, and any special features that make it ideal for families, groups, or individuals."


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
print("Nulls after handling:\n",df.isnull().sum())
# endregion


# region Flooring Columns
#NOTE:For the numeric columns `bathrooms`, `bedrooms`, and `beds`, 
#we applied a floor operation to round down the values as fractional values for those columns don't have meaningful interpretations.

numeric_columns = ['bathrooms', 'bedrooms', 'beds']

for col in numeric_columns:
    df[col] = np.floor(df[col]).astype(int)
# endregion 
# region Text Processing
# region 1) Text Cleaning
print("BEFORE CLEANING:",df['summary'].iloc[8722])
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
    
print("AFTER CLEANING:",df['summary'].iloc[8722])
# endregion
# region 2) Handling Contractions
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

for col in text_columns:
    df[col] = df[col].apply(expand_contractions)
print("AFTER EXPANDING CONTRADICTIONS:",df['summary'].iloc[8722])
# endregion

# region 3) Tokenization
from nltk.tokenize import word_tokenize

for col in text_columns:
    df[col] = df[col].apply(word_tokenize)
# endregion
# region 4) Removing Stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]

for col in text_columns:
    df[col] = df[col].apply(remove_stop_words)
# endregion



# region 5) Lemmatization
from nltk.stem import WordNetLemmatizer
print("BEFORE LEMMATIZATION:",df.iloc[:5,5].to_frame())
lemmatizer = WordNetLemmatizer()

def lemma(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


for col in text_columns:
    df[col] = df[col].apply(lemma)
print("AFTER LEMMATIZATION:",df.iloc[:5,3].to_frame())
# endregion
# region 6) Text Vectorization using Tfidf
   
from sklearn.feature_extraction.text import TfidfVectorizer

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

# endregion

# region 7) Dimensionality Reduction using SVD
from sklearn.decomposition import TruncatedSVD

svd_summary = TruncatedSVD(n_components=35, random_state=0)
svd_space = TruncatedSVD(n_components=35, random_state=0)
svd_transit = TruncatedSVD(n_components=35, random_state=0)
svd_access = TruncatedSVD(n_components=35, random_state=0)
svd_description = TruncatedSVD(n_components=35, random_state=0)
svd_notes = TruncatedSVD(n_components=35, random_state=0)
svd_house_rules = TruncatedSVD(n_components=35, random_state=0)

summary_svd_df = pd.DataFrame(svd_summary.fit_transform(summary_features), columns=[f"summary_svd_{i}" for i in range(35)])
space_svd_df = pd.DataFrame(svd_space.fit_transform(space_features), columns=[f"space_svd_{i}" for i in range(35)])
transit_svd_df = pd.DataFrame(svd_transit.fit_transform(transit_features), columns=[f"transit_svd_{i}" for i in range(35)])
access_svd_df = pd.DataFrame(svd_access.fit_transform(access_features), columns=[f"access_svd_{i}" for i in range(35)])
description_svd_df = pd.DataFrame(svd_description.fit_transform(description_features), columns=[f"description_svd_{i}" for i in range(35)])
notes_svd_df = pd.DataFrame(svd_notes.fit_transform(notes_features), columns=[f"notes_svd_{i}" for i in range(35)])
house_rules_svd_df = pd.DataFrame(svd_house_rules.fit_transform(house_rules_features), columns=[f"house_rules_svd_{i}" for i in range(35)])


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
# endregion
# endregion

# region Sentiment Analysis
# Calculating sentiment score using TextBlob
from textblob import TextBlob
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

for column in ['interaction',  'host_about','neighborhood_overview','name']:
    df[column] = df[column].apply(get_sentiment)

print(df[['interaction','host_about','neighborhood_overview']].head(5))

# Standardizing Short Text Columns
columns_to_standardize = [
    'host_name', 'host_location', 'host_neighbourhood',
    'street', 'neighbourhood', 'neighbourhood_cleansed',
    'city', 'state', 'market', 'smart_location', 'country_code','country','property_type','room_type'
]
for col in columns_to_standardize:
    df[col] = df[col].str.lower().str.strip()
# endregion

#region Encoding
# region 1-Label Encoder
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

# region 2-Binary Mapping of Boolean Columns
columns_to_map = [
    'host_has_profile_pic', 'host_identity_verified', 'is_location_exact',
    'requires_license', 'instant_bookable', 'is_business_travel_ready',
    'require_guest_phone_verification', 'host_is_superhost'
]
mapping_dict = {'t': 1, 'f': 0}
for col in columns_to_map:
    df[col] = df[col].map(mapping_dict)
    
#NOTE:the `require_guest_profile_picture` column is transformed by reversing the typical boolean mapping, where `'t'` (true) is mapped to `0` and `'f'` (false) is mapped to `1`. 
# This reversed mapping was chosen because we found it better not to ask for a guest profile picture
df['require_guest_profile_picture'] = df['require_guest_profile_picture'].map({'t': 0, 'f': 1})
# endregion
#region 3-Mapping Categorical Data to Numeric Values with Rank (Higher is Better)
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


# region Handling Outliers using IQR method
outliers_cols = [
    'host_total_listings_count',
    "latitude", "longitude",
    "accommodates", "bathrooms", "bedrooms", "beds",
    "guests_included", "minimum_nights",
    "maximum_nights", "number_of_reviews", "number_of_stays",'review_scores_rating','host_listings_count'
]
for col in outliers_cols:
  q1 = np.percentile(df[col], 25)
  q3 = np.percentile(df[col], 75)
  norm_range = (q3 - q1) * 1.5
  lower_outliers = df[df[col] < (q1 - norm_range)]
  upper_outliers = df[df[col] > (q3 + norm_range)]
  outliers = len(lower_outliers)+len(upper_outliers)
  print(f"The number of outliers in {col} is : {outliers}\n")
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

for col in outliers_cols:
  q1 = np.percentile(df[col], 25)
  q3 = np.percentile(df[col], 75)
  norm_range = (q3 - q1) * 1.5
  df[col] = np.where(df[col] < (q1 - norm_range), q1 - norm_range, df[col])
  df[col] = np.where(df[col] > (q3 + norm_range), q3 + norm_range, df[col])
for col in outliers_cols:
  q1 = np.percentile(df[col], 25)
  q3 = np.percentile(df[col], 75)
  norm_range = (q3 - q1) * 1.5
  lower_outliers = df[df[col] < (q1 - norm_range)]
  upper_outliers = df[df[col] > (q3 + norm_range)]
  outliers = len(lower_outliers)+len(upper_outliers)
  print(f"AFTER HANDLING OUTLIERS: The number of outliers in {col} is : {outliers}")
# endregion 

# region Feature Engineering
  # region 1)Web Scraping
      #1- Using beautiful soup
       #NOTE:We extracted host ratings from `host_url` column and stores the ratings in the `host_rating` column of and saved it in main csv file
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
      #2- Using Selenium
         #NOTE:We extracted guest favourite labels from `listing_url` column and stores the ratings in the `guest_favorite` column of and 
         #Saved it in main csv file
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
#NOTE:We cleaned and parsed the `amenities` column, extracted all individual items, and grouped them into meaningful categories (e.g., Essentials, Safety, Luxury). For each listing, 
# we counted how many amenities belong to each category and added these counts as new features to the dataset.

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
from collections import defaultdict
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
print("AFTER DIVIDING AMENTIES INTO CATEGORIES:\n",df[['amenities','Essentials','Safety','Luxury','Accessibility','Outdoor','Child & Family-Friendly','Entertainment','Home Appliances']].head(5))
print(df.shape)
# endregion
# region Extracting New features from date columns
  # 1-Extracting number of active years from host since column
from datetime import datetime

df['host_since'] = pd.to_datetime(df['host_since'], format='%m/%d/%Y')

today = pd.to_datetime(datetime.today())
df['years_active'] = (today - df['host_since']).dt.days / 365

df['years_active'] = df['years_active'].round(1)

print(df[['years_active','host_since']].head())
# 2- Extracting review frequency per day from last_review,first_review and number_of_reviews columns
df['first_review'] = pd.to_datetime(df['first_review'], format='%m/%d/%Y')
df['last_review'] = pd.to_datetime(df['last_review'], format='%m/%d/%Y')

df['reviews_per_day'] = df.apply(
    lambda row: row['number_of_reviews'] / ((row['last_review'] - row['first_review']).days)
    if (row['last_review'] - row['first_review']).days != 0 else 0,
    axis=1
)

df['reviews_per_day'] = df['reviews_per_day'].round(4)

print(df[['reviews_per_day', 'first_review', 'last_review', 'number_of_reviews']].head())
    #3-Dividing host location column into host city , host state and host country
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

print(df[['host_location', 'host_city', 'host_state', 'host_country']].head())
 
le = LabelEncoder()
columns_to_encode = ['host_city', 'host_state', 'host_country']

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])

print(df[['host_location', 'host_city', 'host_state', 'host_country']].head())
# endregion
# region Calculating Total cost from price per stay, cleaning fee and security deposit columns
df['total_cost'] = (
    df['price_per_stay'] +
    df['cleaning_fee'] +
    df['security_deposit']
)
print(df[['price_per_stay', 'cleaning_fee', 'security_deposit','total_cost']].head())
# endregion
# endregion

# region Feature Selection
   #  Dropping Unnecessary Columns
df = df.drop(['id','listing_url','host_url','thumbnail_url','host_acceptance_rate','square_feet','summary','space','transit','access','host_since','host_location',
              'host_listings_count','amenities','first_review','last_review','description','notes','house_rules'], axis=1)
#----------------------------------------------------------------------------------------------------------------------------#
#  region Correlation
correlations = df.corr()['review_scores_rating'].abs().drop('review_scores_rating').sort_values(ascending=False)
selected_features = correlations.head(35).index.tolist()

for col in selected_features:   
    print(f"{col}: {round(correlations[col], 4)}")
import plotly.graph_objects as go
correlation_data = df[selected_features].corr()
fig = go.Figure(data=go.Heatmap(
    z=correlation_data.values,
    x=correlation_data.columns,
    y=correlation_data.columns,
    colorscale=['#E41D53', '#EE2D6B', '#F76EA0', '#40E0D0'],
    colorbar=dict(title='Correlation'),
))
fig.update_layout(
    title='Correlation Heatmap for Selected Features',
    xaxis_title='Features',
    yaxis_title='Features',
    xaxis=dict(tickmode='array', tickvals=list(range(len(correlation_data.columns))), ticktext=correlation_data.columns),
    yaxis=dict(tickmode='array', tickvals=list(range(len(correlation_data.columns))), ticktext=correlation_data.columns),
    width=1100, height=800
)
fig.show()
# endregion 
# endregion
# region Model Training and Selection
 # Splitting the data into features and target
x=df[selected_features]
y= df['review_scores_rating']
 # Splitting the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
 # Feature Scaling using Robust Scaler
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
 # Regression Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,root_mean_squared_error
def regression_metrics(y_true, y_pred):
    metrics = {
        'R2 Score': r2_score(y_true, y_pred),
        'Mean Absolute Error (MAE)': mean_absolute_error(y_true, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_true, y_pred),
        'Root Mean Squared Error (RMSE)': root_mean_squared_error(y_true, y_pred),
        'Mean Absolute Percentage Error (MAPE)': mean_absolute_percentage_error(y_true, y_pred)
    }
    return metrics
#-------------------------- LINEAR MODELS------------------------#

#region I)Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

linear_reg = LinearRegression()
cv_scores = cross_val_score(linear_reg, x, y, cv=5, scoring='r2', n_jobs=-1)
print("\nLINEAR REGRESSION MODEL: ")
print(f"Cross-Validation R2 Scores: {cv_scores}")
print(f"Average Cross-Validation R2 Score: {cv_scores.mean():.4f}")
linear_reg.fit(X_train, y_train)
train_r2 = linear_reg.score(X_train, y_train)
print(f"Training R2 Score: {train_r2:.4f}")
y_pred_linear = linear_reg.predict(X_test)
results = regression_metrics(y_test, y_pred_linear)
print("Test Set Metrics:\n")
for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion
#region II) Lasso Regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

param_grid_lasso = {
    'alpha': np.logspace(-6, -3, 10)
}
lasso = Lasso()
grid_search_lasso = GridSearchCV(estimator=lasso, param_grid=param_grid_lasso, cv=5, scoring='r2', n_jobs=-1)
grid_search_lasso.fit(X_train, y_train)
best_params_lasso = grid_search_lasso.best_params_
print("LASSO REGRESSION MODEL: ")
print(f"Best parameters (Lasso): {best_params_lasso}")
best_score_lasso = grid_search_lasso.best_score_
print(f"Best CV score (R²) (Lasso): {best_score_lasso:.4f}")
best_lasso_model = grid_search_lasso.best_estimator_
train_r2 = best_lasso_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2:.4f}")
y_pred_lasso = best_lasso_model.predict(X_test)
results_lasso = regression_metrics(y_test, y_pred_lasso)
print("Test Set Metrics (Lasso):\n")
for metric_name, value in results_lasso.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion

# region III) Ridge Regression
from sklearn.linear_model import Ridge

param_grid_ridge = {
   'alpha': np.logspace(-3, 3, 10) 
}
ridge = Ridge()
grid_search_ridge = GridSearchCV(estimator=ridge, param_grid=param_grid_ridge, cv=5, scoring='r2', n_jobs=-1)
grid_search_ridge.fit(X_train, y_train)
best_params_ridge = grid_search_ridge.best_params_
print("RIDGE REGRESSION MODEL: ")
print(f"Best parameters (Ridge): {best_params_ridge}")
best_score_ridge = grid_search_ridge.best_score_
print(f"Best CV score (R²) (Ridge): {best_score_ridge:.4f}")
best_ridge_model = grid_search_ridge.best_estimator_
train_r2_ridge = best_ridge_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_ridge:.4f}")
y_pred_ridge = best_ridge_model.predict(X_test)
results_ridge = regression_metrics(y_test, y_pred_ridge)
print("Test Set Metrics (Ridge):\n")
for metric_name, value in results_ridge.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion
# region IV) Elastic Net
from sklearn.linear_model import ElasticNet

param_grid_elasticnet = {
   'alpha': np.logspace(-3, 3, 10),  
   'l1_ratio': np.linspace(0, 1, 10)
}
elasticnet = ElasticNet()
grid_search_elasticnet = GridSearchCV(estimator=elasticnet, param_grid=param_grid_elasticnet, cv=5, scoring='r2', n_jobs=-1)
grid_search_elasticnet.fit(X_train, y_train)
best_params_elasticnet = grid_search_elasticnet.best_params_
print("ELASTIC NET MODEL: ")
print(f"Best parameters (ElasticNet): {best_params_elasticnet}")
best_score_elasticnet = grid_search_elasticnet.best_score_
print(f"Best CV score (R²) (ElasticNet): {best_score_elasticnet:.4f}")
best_elasticnet_model = grid_search_elasticnet.best_estimator_
train_r2_elasticnet = best_elasticnet_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_elasticnet:.4f}")
y_pred_elasticnet = best_elasticnet_model.predict(X_test)
results_elasticnet = regression_metrics(y_test, y_pred_elasticnet)
print("Test Set Metrics (ElasticNet):\n")
for metric_name, value in results_elasticnet.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion

# region V) Support Vector Regression (SVR)

from sklearn.svm import SVR

param_grid_svr = {
    'C': np.logspace(-1, 2, 5),
    'epsilon': np.linspace(0.01, 1, 5),
    'kernel': ['rbf']
}
svr = SVR()
grid_search_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr, cv=5, scoring='r2', n_jobs=-1)
grid_search_svr.fit(X_train, y_train)
best_params_svr = grid_search_svr.best_params_
print("SVR MODEL: ")
print(f"Best parameters (SVR): {best_params_svr}")
best_score_svr = grid_search_svr.best_score_
print(f"Best CV score (R²) (SVR): {best_score_svr:.4f}")
best_svr_model = grid_search_svr.best_estimator_
train_r2_svr = best_svr_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_svr:.4f}")
y_pred_svr = best_svr_model.predict(X_test)
results_svr = regression_metrics(y_test, y_pred_svr)
print("Test Set Metrics (SVR):\n")
for metric_name, value in results_svr.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion

#Ensemble models
 #1) Bagging models
   # region 1-Random Forest 
from sklearn.ensemble import RandomForestRegressor

param_grid_rf = {
   'n_estimators': [50, 100],  
   'max_depth': [None, 10],  
   'min_samples_split': [2, 5], 
   'min_samples_leaf': [1, 2]  
}
rf = RandomForestRegressor(random_state=0)
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, scoring='r2', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_params_rf = grid_search_rf.best_params_
print("RANDOM FOREST MODEL: ")
print(f"Best parameters (RandomForest): {best_params_rf}")
best_score_rf = grid_search_rf.best_score_
print(f"Best CV score (R²) (RandomForest): {best_score_rf:.4f}")
best_rf_model = grid_search_rf.best_estimator_
train_r2_rf = best_rf_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_rf:.4f}")
y_pred_rf = best_rf_model.predict(X_test)
results_rf = regression_metrics(y_test, y_pred_rf)
print("Test Set Metrics (RandomForest):\n")
for metric_name, value in results_rf.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion

# region 2-Decision tree
from sklearn.tree import DecisionTreeRegressor
param_grid_dt = {
   'max_depth': [None, 10, 20],  
   'min_samples_split': [2, 5], 
   'min_samples_leaf': [1, 2]  
}
dt = DecisionTreeRegressor(random_state=0)
grid_search_dt = GridSearchCV(estimator=dt, param_grid=param_grid_dt, cv=5, scoring='r2', n_jobs=-1)
grid_search_dt.fit(X_train, y_train)
best_params_dt = grid_search_dt.best_params_
print("DESICION TREE MODEL: ")
print(f"Best parameters (DecisionTree): {best_params_dt}")
best_score_dt = grid_search_dt.best_score_
print(f"Best CV score (R²) (DecisionTree): {best_score_dt:.4f}")
best_dt_model = grid_search_dt.best_estimator_
train_r2_dt = best_dt_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_dt:.4f}")
y_pred_dt = best_dt_model.predict(X_test)
results_dt = regression_metrics(y_test, y_pred_dt)
print("Test Set Metrics (DecisionTree):\n")
for metric_name, value in results_dt.items():
    print(f"{metric_name}: {value:.4f}")
# endregion

#-------------------------------------------#
#2) Boosting models
# region XGBoost
from xgboost import XGBRegressor

xgb_model = XGBRegressor(
    n_estimators=105,
    max_depth=3,
    learning_rate=0.1435307406386383,
    subsample=0.955315863949093,
    colsample_bytree=0.9788141815588388,
    random_state=0,
    verbosity=0
)
cv_scores_xgb = cross_val_score(xgb_model, x, y, cv=5, scoring='r2', n_jobs=-1)
print(f"Cross-Validation R² Scores: {cv_scores_xgb}")
print(f"Average Cross-Validation R² Score: {cv_scores_xgb.mean():.4f}")
xgb_model.fit(X_train, y_train)
train_r2_xgb = xgb_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_xgb:.4f}")
y_pred_xgb = xgb_model.predict(X_test)
results_xgb = regression_metrics(y_test, y_pred_xgb)
print("Test Set Metrics (XGBoost):")
for metric_name, value in results_xgb.items():
    print(f"{metric_name}: {value:.4f}")

print("\n")
# endregion

# region CatBoost
from catboost import CatBoostRegressor


cat_model = CatBoostRegressor(depth=5, 
                                   iterations=3234, 
                                   learning_rate=0.009991409541444202, 
                                   l2_leaf_reg=64, 
                                   bagging_temperature= 0.9446579587709355, 
                                   random_state=0, verbose=0)
cv_scores_cat = cross_val_score(cat_model, x, y, cv=5, scoring='r2', n_jobs=-1)
print(f"Cross-Validation R² Scores: {cv_scores_cat}")
print(f"Average Cross-Validation R² Score: {cv_scores_cat.mean():.4f}")
cat_model.fit(X_train, y_train)
train_r2_cat = cat_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_cat:.4f}")
y_pred_cat = cat_model.predict(X_test)
results_cat = regression_metrics(y_test, y_pred_cat)
print("Test Set Metrics (CatBoost):")
for metric_name, value in results_cat.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion

# region Light gbm
from lightgbm import LGBMRegressor

light_model = LGBMRegressor(
    n_estimators=1547,
    learning_rate=0.014286813901088678,
    max_depth=3,
    reg_lambda=35,
    subsample=0.8148197144286347,
    subsample_freq=5,
    random_state=0
)
cv_scores_light = cross_val_score(light_model, x, y, cv=5, scoring='r2', n_jobs=-1)
print(f"Cross-Validation R² Scores: {cv_scores_light}")
print(f"Average Cross-Validation R² Score: {cv_scores_light.mean():.4f}")
light_model.fit(X_train, y_train)
train_r2_light = light_model.score(X_train, y_train)
print(f"Training R² Score: {train_r2_light:.4f}")
y_pred_light = light_model.predict(X_test)
results_light = regression_metrics(y_test, y_pred_light)
print("Test Set Metrics (LightGBM):")
for metric_name, value in results_light.items():
    print(f"{metric_name}: {value:.4f}")

print("\n")
# endregion 
# region 3) Stacking model
  #Stacking Regressor Model with CatBoost, LightGBM, and XGBoost
from sklearn.ensemble import StackingRegressor

lgb_model = LGBMRegressor(
    random_state=0,
    n_estimators=1534,
    learning_rate=0.010382774494101006,
    max_depth=3,
    reg_lambda=35,
    subsample=0.8148197144286347,
    subsample_freq=5
)
cat_model = CatBoostRegressor(
    verbose=0,
    random_state=0,
    depth=5,
    iterations=3234,
    learning_rate=0.009991409541444202,
    l2_leaf_reg=64,
    bagging_temperature=0.9446579587709355
)
meta_model = XGBRegressor(
    n_estimators=35,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=0.8,
    random_state=0,
    verbosity=1
)


stacked_model3 = StackingRegressor(
    estimators=[
        ('lightgbm', lgb_model),
        ('catboost', cat_model)
    ],
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=True 
)

stacked_model3.fit(X_train, y_train)
y_pred_stack3 = stacked_model3.predict(X_test)
train_stack = stacked_model3.score(X_train, y_train)
print(f"Training R² Score: {train_stack:.4f}")
results_stacking = regression_metrics(y_test,y_pred_stack3)
print("Test Set Metrics (Stacking):")
for metric_name, value in results_stacking.items():
    print(f"{metric_name}: {value:.4f}")
print("\n")
# endregion

# region Model Accuracy Comparison
results = {
    'Linear Regression': regression_metrics(y_test, y_pred_linear),
    'Lasso Regression': regression_metrics(y_test, y_pred_lasso),
    'Ridge Regression': regression_metrics(y_test, y_pred_ridge),
    'ElasticNet': regression_metrics(y_test, y_pred_elasticnet),
    'SVR': regression_metrics(y_test, y_pred_svr),
    'Random Forest': regression_metrics(y_test, y_pred_rf),
    'Decision Tree': regression_metrics(y_test, y_pred_dt),
    'XGBoost': regression_metrics(y_test, y_pred_xgb),
    'CatBoost': regression_metrics(y_test, y_pred_cat),
    'LightGBM': regression_metrics(y_test, y_pred_light),
    'Stacking': regression_metrics(y_test, y_pred_stack3)
}
#R2 SCORE
r2_scores = {model: round(results[model]['R2 Score'], 4) for model in results}
r2_table = pd.DataFrame(list(r2_scores.items()), columns=['Model', 'R² Score'])
print("R² Table:")
print(r2_table)

plt.figure(figsize=(10, 6))
plt.bar(r2_scores.keys(), r2_scores.values(), color='#40E0D0')
plt.title('Model R² Scores', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#MEAN SQUARED ERROR
mse_scores = {model: round(results[model]['Mean Squared Error (MSE)'], 4) for model in results}
mse_table = pd.DataFrame(list(mse_scores.items()), columns=['Model', 'Mean Squared Error (MSE)'])
print("MSE Table:")
print(mse_table)
# endregion
plt.figure(figsize=(10, 6))
plt.bar(mse_scores.keys(), mse_scores.values(), color='#E41D53')
plt.title('Model Mean Squared Error (MSE)', fontsize=14)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Mean Squared Error', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()