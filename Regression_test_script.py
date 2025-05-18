import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,root_mean_squared_error

df = pd.read_csv('regressionTestFirst20.csv')

summary_tfidf = joblib.load("regression_models/summary_tfidf_vectorizer")
space_tfidf = joblib.load("regression_models/space_tfidf_vectorizer")
description_tfidf = joblib.load("regression_models/description_tfidf_vectorizer")
notes_tfidf = joblib.load("regression_models/notes_tfidf_vectorizer")
transit_tfidf = joblib.load("regression_models/transit_tfidf_vectorizer")
access_tfidf = joblib.load("regression_models/access_tfidf_vectorizer")
house_rules_tfidf = joblib.load("regression_models/house_rules_tfidf_vectorizer")

svd_summary = joblib.load("regression_models/svd_summary_model")
svd_space = joblib.load("regression_models/svd_space_model")
svd_transit = joblib.load("regression_models/svd_transit_model")
svd_access = joblib.load("regression_models/svd_access_model")
svd_description = joblib.load("regression_models/svd_description_model")
svd_notes = joblib.load("regression_models/svd_notes_model")
svd_house_rules = joblib.load("regression_models/svd_house_rules_model")

label_encoders=joblib.load('regression_models/label_encoders')
scaler = joblib.load('regression_models/scaler')
model = joblib.load('regression_models/stacked_model')


#-------------------------------------------------------------------------------------------------------------------
#nulls
df['property_type'] = df['property_type'].fillna('House')
df['room_type'] = df['room_type'].fillna('Entire home/apt')
df['host_rating'] = 4.8
df['guest_favorite'] = 0
df['host_is_superhost']=df['host_is_superhost'].fillna('f')
df['instant_bookable']=df['instant_bookable'].fillna('t')
df['is_location_exact']=df['is_location_exact'].fillna('f')
df['accommodates']=df['accommodates'].fillna(4)
df['minimum_nights']=df['minimum_nights'].fillna(2)
df['beds']=df['beds'].fillna(2)
df['cancellation_policy']=df['cancellation_policy'].fillna('strict_14_with_grace_period')
df['host_total_listings_count']=df['host_total_listings_count'].fillna(2.0)
df['amenities'] = df['amenities'].fillna('[]') 
df['cleaning_fee'] = df['cleaning_fee'].fillna('$0.0')
df['maximum_nights'] = df['maximum_nights'].fillna(365)


#-------------------------------------------------------------------------------------------------------------------
# label encoding
columns_to_encode = [
    'property_type', 'room_type'
]
for col in columns_to_encode:
    df[col] = df[col].str.lower().str.strip()
for col in columns_to_encode:
    if col in df.columns:
        le = label_encoders[col]
        df[col] = df[col].apply(lambda x: le.transform([x])[0]) 

#-------------------------------------------------------------------------------------------------------------------
# binary columns

binary_columns = ['host_is_superhost', 'instant_bookable', 'is_location_exact']
mapping_dict = {'t': 1, 'f': 0}
for col in binary_columns:
    df[col] = df[col].map(mapping_dict)


#-------------------------------------------------------------------------------------------------------------------
# cancellation policy

cancellation_rank = {
    'flexible': 5,
    'moderate': 4,
    'strict': 3,
    'strict_14_with_grace_period': 2,
    'super_strict_30': 1,
    'super_strict_60': 0
}
df['cancellation_policy'] = df['cancellation_policy'].map(cancellation_rank)

#-------------------------------------------------------------------------------------------------------------------
# text preprocessing


text_columns = ['summary', 'space', 'description','notes', 'transit', 'access', 'house_rules']

for col in text_columns:
    df[col] = df[col].fillna("No information Provided") 
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

for col in text_columns:
    df[col] = df[col].apply(word_tokenize)
stop_words = set(stopwords.words('english'))
def remove_stop_words(tokens):
    return [word for word in tokens if word not in stop_words]
for col in text_columns:
    df[col] = df[col].apply(remove_stop_words)
lemmatizer = WordNetLemmatizer()
def lemma(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]
for col in text_columns:
    df[col] = df[col].apply(lemma)

for col in text_columns:
    df[col] = df[col].apply(lambda x: ' '.join(x))

def tfidf_svd_transform(text_series, tfidf_model, svd_model, prefix):
    tfidf_matrix = tfidf_model.transform(text_series) 
    svd_matrix = svd_model.transform(tfidf_matrix)
    svd_df = pd.DataFrame(svd_matrix, columns=[f"{prefix}_svd_{i}" for i in range(svd_matrix.shape[1])])
    return svd_df

summary_svd = tfidf_svd_transform(df['summary'], summary_tfidf, svd_summary, 'summary')
space_svd = tfidf_svd_transform(df['space'], space_tfidf,  svd_space, 'space')
transit_svd = tfidf_svd_transform(df['transit'], transit_tfidf,  svd_transit, 'transit')
access_svd = tfidf_svd_transform(df['access'], access_tfidf,  svd_access, 'access')
description_svd = tfidf_svd_transform(df['description'], description_tfidf,  svd_description, 'description')
notes_svd = tfidf_svd_transform(df['notes'], notes_tfidf,  svd_notes, 'notes')
houserules_svd = tfidf_svd_transform(df['house_rules'], house_rules_tfidf, svd_house_rules, 'house_rules')

df = pd.concat(
    [df, summary_svd, space_svd, transit_svd, access_svd, description_svd, notes_svd, houserules_svd], 
    axis=1
)

df.drop(columns=[
    'summary', 'space', 'transit', 'access',
    'description', 'notes', 'house_rules'
], inplace=True)

#-------------------------------------------------------------------------------------------------------------------

df['beds'] = np.floor(df['beds']).astype(int)
df['cleaning_fee'] = df['cleaning_fee'].str.replace(',', '').str.replace('$', '').astype(float)

#-------------------------------------------------------------------------------------------------------------------
# sentiment analysis

def get_sentiment(text):
    return TextBlob(text).sentiment.polarity
 
sentiment_Cols=['neighborhood_overview']

for column_sent in sentiment_Cols:
   df[column_sent] = df[column_sent].fillna("No information Provided") 

for column_sent in sentiment_Cols:
    df[column_sent] = df[column_sent].apply(get_sentiment)

#-------------------------------------------------------------------------------------------------------------------


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


#-------------------------------------------------------------------------------------------------------------------

features = [
    "host_rating",
    "host_is_superhost",
    "host_total_listings_count",
    "cancellation_policy",
    "Safety",
    "guest_favorite",
    "house_rules_svd_15",
    "Outdoor",
    "summary_svd_10",
    "house_rules_svd_5",
    "instant_bookable",
    "transit_svd_1",
    "maximum_nights",
    "summary_svd_8",
    "house_rules_svd_3",
    "Luxury",
    "transit_svd_0",
    "Essentials",
    "notes_svd_3",
    "access_svd_1",
    "description_svd_12",
    "accommodates",
    "description_svd_3",
    "property_type",
    "Home Appliances",
    "minimum_nights",
    "beds",
    "house_rules_svd_4",
    "cleaning_fee",
    "is_location_exact",
    "room_type",
    "access_svd_0",
    "access_svd_4",
    "space_svd_10",
    "neighborhood_overview"
]

X = df[features]
y = df['review_scores_rating']



X = scaler.transform(X)
def regression_metrics(y_true, y_pred):
    metrics = {
        'R2 Score': r2_score(y_true, y_pred),
        'Mean Squared Error (MSE)': mean_squared_error(y_true, y_pred)
    }
    return metrics

y_pred =model.predict(X)

results = regression_metrics(y, y_pred)
print("Test Set Metrics:")
for metric_name, value in results.items():
    print(f"{metric_name}: {value:.4f}")


results_df = pd.DataFrame({
    'Actual': y,
    'Predicted': y_pred
})
results_df.to_csv('results_regression.csv', index=False)
print("Results saved to results_regression.csv")

