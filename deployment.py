import streamlit as st
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import joblib
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

st.set_page_config(page_title='Guest Satisfaction Prediction', page_icon='images/logo.png', initial_sidebar_state='expanded')

regression_df = pd.read_csv('GuestSatisfactionPrediction.csv')
classification_df = pd.read_csv('GuestSatisfactionPredictionMilestone2.csv')

st.sidebar.markdown("### Menu")
page = st.sidebar.radio("Navigation", ["Home", "Prediction With CSV file", "Prediction Form","Metrics"])

if page == "Home":
    st.title("Welcome to the Airbnb Guest Satisfaction Prediction App!")
    st.image("images/Airbnb.png",width=500)
    st.markdown("""
## 🏡 Airbnb Guest Satisfaction Prediction

This project aims to assist Airbnb hosts in **predicting the satisfaction level of guests** booking their listings by leveraging **machine learning techniques**. The goal is to build a model that can **anticipate guest satisfaction** based on various **listing and host features**, such as location, amenities, host behavior, and property attributes.

The model is trained on a dataset collected from the Airbnb platform, with the target variable being either:
- `review_scores_rating` (for **regression**) to predict a continuous satisfaction score, or
- `guest_satisfaction` (for **classification**) to categorize satisfaction into three levels: **Very High**, **High**, and **Average**.

By analyzing key factors influencing guest experiences, this tool provides insights to help hosts **enhance their listings and overall guest satisfaction**.
""")
    regression_data_dict = pd.DataFrame({
        'Column': [
            'id', 'listing_url', 'name', 'summary', 'space', 'description',
            'neighborhood_overview', 'notes', 'transit', 'access', 'interaction',
            'house_rules', 'thumbnail_url', 'host_id', 'host_url', 'host_name',
            'host_since', 'host_location', 'host_about', 'host_response_time',
            'host_response_rate', 'host_acceptance_rate', 'host_is_superhost',
            'host_neighbourhood', 'host_listings_count', 'host_total_listings_count',
            'host_has_profile_pic', 'host_identity_verified', 'street',
            'neighbourhood', 'neighbourhood_cleansed', 'city', 'state', 'zipcode',
            'market', 'smart_location', 'country_code', 'country', 'latitude',
            'longitude', 'is_location_exact', 'property_type', 'room_type',
            'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities',
            'square_feet', 'nightly_price', 'price_per_stay', 'security_deposit',
            'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
            'maximum_nights', 'number_of_reviews', 'number_of_stays', 'first_review',
            'last_review', 'review_scores_rating', 'requires_license',
            'instant_bookable', 'is_business_travel_ready', 'cancellation_policy',
            'require_guest_profile_picture', 'require_guest_phone_verification'
        ],
        'Description': [
            "Unique identifier for the listing.", "URL to the Airbnb listing.", "Title of the listing.",
            "Short summary description provided by the host.", "Description of the space guests can access.",
            "Full description of the listing.", "Overview of the neighborhood.",
            "Additional notes from the host.", "Information about nearby transit options.",
            "Details on guest access to the property.", "Information on how hosts interact with guests.",
            "House rules set by the host.", "Thumbnail image URL of the listing.",
            "Unique identifier for the host.", "URL to the host’s profile.", "Name of the host.",
            "Date the host joined Airbnb.", "Location of the host.", "Bio or description provided by the host.",
            "Average time the host takes to respond.", "Percentage of messages responded to.",
            "Rate at which host accepts bookings.", "Whether the host is a superhost.",
            "Host’s self-described neighborhood.", "Number of listings the host has.",
            "Total listings count (including inactive).", "Whether the host has a profile picture.",
            "Whether the host’s identity is verified.", "Street address of the listing.",
            "Name of the local neighborhood.", "Cleaned or standardized neighborhood name.",
            "City of the listing.", "State where the listing is located.", "Zip code of the listing.",
            "General market area.", "Formatted location string.", "Country code.", "Full country name.",
            "Latitude of the listing.", "Longitude of the listing.", "Whether the location is exact.",
            "Type of property (e.g., apartment, house).", "Type of room available.",
            "Number of people the listing can accommodate.", "Number of bathrooms.",
            "Number of bedrooms.", "Number of beds.", "Type of bed.", "List of included amenities.",
            "Size of the listing in square feet.", "Price per night.", "Total price for a stay.",
            "Security deposit amount.", "Cleaning fee charged.", "Number of guests included in base price.",
            "Extra charge for additional guests.", "Minimum nights required per booking.",
            "Maximum nights allowed per booking.", "Total number of reviews.", "Total number of stays.",
            "Date of the first review.", "Date of the most recent review.",
            "Overall review score (target variable).", "Whether a license is required.",
            "Whether guests can book instantly.", "Whether listing is ready for business travelers.",
            "Policy on cancellations.", "Whether guest profile pictures are required.",
            "Whether guest phone verification is required."
        ]
    })

    classification_data_dict = regression_data_dict.copy()
    classification_data_dict.loc[classification_data_dict['Column'] == 'review_scores_rating', 'Column'] = 'guest_satisfaction'
    classification_data_dict.loc[classification_data_dict['Column'] == 'guest_satisfaction', 'Description'] = 'Guest satisfaction level (target variable: Very High, High, Average).'
    with st.expander("📘 Regression Dataset Dictionary (Target: review_scores_rating)"):
        st.dataframe(regression_data_dict, use_container_width=True)
        st.markdown("**Sample data (5 rows) from Regression dataset:**")
        st.dataframe(regression_df.head(), use_container_width=True)

    # Show Classification dictionary and sample data
    with st.expander("📗 Classification Dataset Dictionary (Target: guest_satisfaction)"):
        st.dataframe(classification_data_dict, use_container_width=True)
        st.markdown("**Sample data (5 rows) from Classification dataset:**")
        st.dataframe(classification_df.head(), use_container_width=True)

    st.markdown(f"""
    <style>
        .hover-div-documentation {{
            padding: 10px;
            border-radius: 10px;
            background: linear-gradient(90deg, #E41D53, #EE2D6B, #F76EA0);
            margin-bottom: 10px;
            display: flex;
            justify-content: center;
            align-items: center;
            transition: background 0.3s ease, box-shadow 0.3s ease;
            cursor: pointer;
            text-decoration: none;
            color: white;
            font-weight: bold;
            font-size: 1.1rem;
        }}
        .hover-div-documentation:hover {{
            background: linear-gradient(90deg, #F76EA0, #EE2D6B, #E41D53);
            box-shadow: 0px 4px 15px rgba(228, 29, 83, 0.7);
            color: white;
        }}
        .hover-div-documentation h4 {{
            margin: 0;
            text-align: center;
            width: 100%;
        }}
    </style>

    <a href="https://github.com/TokaKhaled4/Guest_Satisfaction_Prediction_Project/blob/main/FINALREPORT_SC17.pdf" target="_blank" class="hover-div-documentation">
        <h4>View Our Documentation</h4>
    </a>
""", unsafe_allow_html=True)

elif page == "Prediction With CSV file":
    task_type = st.sidebar.selectbox("Select Task Type", ["Regression", "Classification"])

    if task_type == "Regression":
        st.markdown("## Review Scores Rating Prediction")
        st.sidebar.markdown("Upload a CSV file containing Airbnb listing data to predict guest satisfaction or review scores.")

        uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])

        @st.cache_data(show_spinner=False)
        def load_models():
            with st.spinner('Loading models...'):
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

                label_encoders = joblib.load('regression_models/label_encoders')
                scaler = joblib.load('regression_models/scaler')
                model = joblib.load('regression_models/stacked_model')

            return (
                summary_tfidf, space_tfidf, description_tfidf, notes_tfidf, transit_tfidf,
                access_tfidf, house_rules_tfidf, svd_summary, svd_space, svd_transit,
                svd_access, svd_description, svd_notes, svd_house_rules, label_encoders,
                scaler, model
            )

        if uploaded_file:
            st.sidebar.success("CSV file uploaded successfully!")
            models = load_models()
        else:
            st.sidebar.info("Please upload a CSV file to proceed with predictions.")

        def preprocess_text_columns(df, text_columns):
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

            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            for col in text_columns:
                df[col] = df[col].fillna("No information Provided")
                df[col] = (
                    df[col]
                    .str.lower()
                    .str.replace(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", "", regex=True)
                    .str.replace(r"<.*?>", "", regex=True)
                    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                df[col] = df[col].apply(expand_contractions)
                df[col] = df[col].apply(word_tokenize)
                df[col] = df[col].apply(lambda tokens: [w for w in tokens if w not in stop_words])
                df[col] = df[col].apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])
                df[col] = df[col].apply(lambda tokens: ' '.join(tokens))
            return df

        def tfidf_svd_transform(text_series, tfidf_model, svd_model, prefix):
            tfidf_matrix = tfidf_model.transform(text_series)
            svd_matrix = svd_model.transform(tfidf_matrix)
            svd_df = pd.DataFrame(svd_matrix, columns=[f"{prefix}_svd_{i}" for i in range(svd_matrix.shape[1])])
            return svd_df

        def categorize_amenities(amenities_str, amenity_to_category, all_categories):
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

        # def get_sentiment(text):
        #     return TextBlob(text).sentiment.polarity

        def regression_metrics(y_true, y_pred):
            return {
                'R2 Score': r2_score(y_true, y_pred),
                'Mean Squared Error (MSE)': mean_squared_error(y_true, y_pred)
            }


        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            (
                summary_tfidf, space_tfidf, description_tfidf, notes_tfidf, transit_tfidf,
                access_tfidf, house_rules_tfidf, svd_summary, svd_space, svd_transit,
                svd_access, svd_description, svd_notes, svd_house_rules, label_encoders,
                scaler, model
            ) = load_models()

            columns_to_encode = ['property_type', 'room_type']
            for col in columns_to_encode:
                df[col] = df[col].str.lower().str.strip()
                if col in df.columns:
                    le = label_encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

            df['property_type'] = df['property_type'].fillna(23)
            df['room_type'] = df['room_type'].fillna(0)

            binary_columns = ['host_is_superhost', 'instant_bookable', 'is_location_exact']
            mapping_dict = {'t': 1, 'f': 0}
            for col in binary_columns:
                df[col] = df[col].map(mapping_dict)
            df['host_is_superhost'] = df['host_is_superhost'].fillna(0)
            df['instant_bookable'] = df['instant_bookable'].fillna(1)
            df['is_location_exact'] = df['is_location_exact'].fillna(1)

            df['host_rating'] = 4.8
            df['guest_favorite'] = 0

            cancellation_rank = {
                'flexible': 5,
                'moderate': 4,
                'strict': 3,
                'strict_14_with_grace_period': 2,
                'super_strict_30': 1,
                'super_strict_60': 0
            }
            df['cancellation_policy'] = df['cancellation_policy'].map(cancellation_rank)
            df['cancellation_policy'] = df['cancellation_policy'].fillna(2)

            text_columns = ['summary', 'space', 'description', 'notes', 'transit', 'access', 'house_rules']
            df = preprocess_text_columns(df, text_columns)

            # SVD transformations
            summary_svd = tfidf_svd_transform(df['summary'], summary_tfidf, svd_summary, 'summary')
            space_svd = tfidf_svd_transform(df['space'], space_tfidf, svd_space, 'space')
            transit_svd = tfidf_svd_transform(df['transit'], transit_tfidf, svd_transit, 'transit')
            access_svd = tfidf_svd_transform(df['access'], access_tfidf, svd_access, 'access')
            description_svd = tfidf_svd_transform(df['description'], description_tfidf, svd_description, 'description')
            notes_svd = tfidf_svd_transform(df['notes'], notes_tfidf, svd_notes, 'notes')
            houserules_svd = tfidf_svd_transform(df['house_rules'], house_rules_tfidf, svd_house_rules, 'house_rules')

            df = pd.concat(
                [df, summary_svd, space_svd, transit_svd, access_svd, description_svd, notes_svd, houserules_svd],
                axis=1
            )

            df.drop(columns=text_columns, inplace=True)

            df['accommodates'] = df['accommodates'].fillna(4)
            df['minimum_nights'] = df['minimum_nights'].fillna(2)
            df['beds'] = df['beds'].fillna(2)
            df['beds'] = np.floor(df['beds']).astype(int)
            df['host_total_listings_count'] = df['host_total_listings_count'].fillna(2.0)
            df['cleaning_fee'] = df['cleaning_fee'].str.replace(',', '').str.replace('$', '').astype(float)
            df['cleaning_fee'] = df['cleaning_fee'].fillna(0.0)

            # sentiment_cols = ['neighborhood_overview']
            # for col in sentiment_cols:
            #     df[col] = df[col].fillna("No information Provided")
            #     df[col] = df[col].apply(get_sentiment)

            df['amenities'] = df['amenities'].fillna('[]')

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
            

            if 'neighbourhood_cleansed' in df.columns:
                le_nb = label_encoders['neighbourhood_cleansed']
                df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].str.lower().str.strip()
                df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].apply(lambda x: le_nb.transform([x])[0] if x in le_nb.classes_ else -1)

            selected_features = [
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
            
            X = df[selected_features]

            X_scaled = scaler.transform(X)

            y_pred = model.predict(X_scaled)

            if 'review_scores_rating' in df.columns:
                y_true = df['review_scores_rating'].values
                metrics = regression_metrics(y_true, y_pred)
                st.subheader("Regression Metrics")
                
            for k, v in metrics.items():
                text = f"{k}: {v:.4f}"
                if k.lower() in ['r2', 'r_squared']:
                    st.markdown(f"<span style='color: green; font-weight: bold;'>{text}</span>", unsafe_allow_html=True)
                elif k.lower() == 'mse':
                    st.markdown(f"<span style='color: orange; font-weight: bold;'>{text}</span>", unsafe_allow_html=True)
                else:
                    st.write(text)

    else:
        st.markdown("## Guest Satisfaction Prediction")
        st.sidebar.markdown("Upload a CSV file containing Airbnb listing data to predict guest satisfaction or review scores.")

        uploaded_file = st.file_uploader("Choose your CSV file", type=["csv"])

        @st.cache_data(show_spinner=False)
        def load_models():
            with st.spinner('Loading models...'):
                transit_tfidf = joblib.load("classification_models/transit_tfidf_vectorizer")
                house_rules_tfidf = joblib.load("classification_models/house_rules_tfidf_vectorizer")
                svd_transit = joblib.load("classification_models/svd_transit_model")
                svd_house_rules = joblib.load("classification_models/svd_house_rules_model")

                label_encoders=joblib.load('classification_models/label_encoders')
                scaler = joblib.load('classification_models/scaler')
                model = joblib.load('classification_models/model')

            return (
                transit_tfidf,house_rules_tfidf,svd_transit,svd_house_rules, label_encoders,
                scaler, model
            )

        if uploaded_file:
            st.sidebar.success("CSV file uploaded successfully!")
            models = load_models()
        else:
            st.sidebar.info("Please upload a CSV file to proceed with predictions.")

        def preprocess_text_columns(df, text_columns):
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

            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            for col in text_columns:
                df[col] = df[col].fillna("No information Provided")
                df[col] = (
                    df[col]
                    .str.lower()
                    .str.replace(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", "", regex=True)
                    .str.replace(r"<.*?>", "", regex=True)
                    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                df[col] = df[col].apply(expand_contractions)
                df[col] = df[col].apply(word_tokenize)
                df[col] = df[col].apply(lambda tokens: [w for w in tokens if w not in stop_words])
                df[col] = df[col].apply(lambda tokens: [lemmatizer.lemmatize(w) for w in tokens])
                df[col] = df[col].apply(lambda tokens: ' '.join(tokens))
            return df

        def tfidf_svd_transform(text_series, tfidf_model, svd_model, prefix):
            tfidf_matrix = tfidf_model.transform(text_series)
            svd_matrix = svd_model.transform(tfidf_matrix)
            svd_df = pd.DataFrame(svd_matrix, columns=[f"{prefix}_svd_{i}" for i in range(svd_matrix.shape[1])])
            return svd_df
 
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            (
                transit_tfidf,house_rules_tfidf,svd_transit,svd_house_rules, label_encoders,
                scaler, model
            ) = load_models()
            df['host_rating'] = 4.8
            df['guest_favorite'] = 0
            df['number_of_stays']=df['number_of_stays'].fillna(34.0)
            df['amenities'] = df['amenities'].fillna('[]') 
            df['host_is_superhost']=df['host_is_superhost'].fillna('f')
            df['instant_bookable']=df['instant_bookable'].fillna('t')
            df['is_location_exact']=df['is_location_exact'].fillna('f')
            df['cancellation_policy']=df['cancellation_policy'].fillna('strict_14_with_grace_period')
            df['number_of_reviews'] = df['number_of_reviews'].fillna(17.0)
            df['neighbourhood_cleansed']=df['neighbourhood_cleansed'].fillna('Mission Bay')
            df['neighbourhood'] = df['neighbourhood'].fillna('Pacific Beach')
            df['host_neighbourhood'] = df['host_neighbourhood'].fillna('Pacific Beach')
            df['host_name'] = df['host_name'].fillna('SeaBreeze')
            df['host_total_listings_count']=df['host_total_listings_count'].fillna(2.0)
            df['require_guest_phone_verification'] = df['require_guest_phone_verification'].fillna('f')

            columns_to_encode = ['property_type', 'room_type','neighbourhood_cleansed','neighbourhood', 'host_neighbourhood','host_name']
            for col in columns_to_encode:
                df[col] = df[col].str.lower().str.strip()
                if col in df.columns:
                    le = label_encoders[col]
                    df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

            binary_columns = ['host_is_superhost', 'instant_bookable', 'is_location_exact','require_guest_phone_verification']
            mapping_dict = {'t': 1, 'f': 0}
            for col in binary_columns:
                df[col] = df[col].map(mapping_dict)
            cancellation_rank = {
                'flexible': 5,
                'moderate': 4,
                'strict': 3,
                'strict_14_with_grace_period': 2,
                'super_strict_30': 1,
                'super_strict_60': 0
            }
            df['cancellation_policy'] = df['cancellation_policy'].map(cancellation_rank)

            text_columns = ['transit', 'house_rules']
            df = preprocess_text_columns(df, text_columns)


            transit_svd = tfidf_svd_transform(df['transit'], transit_tfidf, svd_transit, 'transit')
            houserules_svd = tfidf_svd_transform(df['house_rules'], house_rules_tfidf, svd_house_rules, 'house_rules')

            df = pd.concat(
                [df,transit_svd, houserules_svd],
                axis=1
            )

            df.drop(columns=text_columns, inplace=True)

            df['amenities'] = df['amenities'].fillna('[]')

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
            
            guest_satisfaction_mapping={
            'Very High':2,
            'High':1,
            'Average':0,
            }
            df['guest_satisfaction'] = df['guest_satisfaction'].map(guest_satisfaction_mapping)
           
            selected_features = [
                'host_rating', 'Luxury', 'Safety', 'property_type', 'number_of_stays',
                'host_is_superhost', 'guest_favorite', 'is_location_exact', 'house_rules_svd_15',
                'number_of_reviews', 'room_type', 'cancellation_policy', 'neighbourhood_cleansed',
                'transit_svd_1', 'neighbourhood', 'Outdoor', 'host_name', 'host_neighbourhood',
                'host_total_listings_count', 'Essentials', 'instant_bookable',
                'require_guest_phone_verification'
            ]
            
            X = df[selected_features]
            y_true = df['guest_satisfaction']
            X = scaler.transform(X)
            y_pred = model.predict(X)
            metrics = accuracy_score(y_true, y_pred)
            st.subheader("Accuracy Score")
            st.write(f"Accuracy: {metrics:.4f}")

elif page == "Prediction Form":
    task_type = st.sidebar.selectbox("Select Task Type", ["Regression", "Classification"])
    if task_type == "Regression":
        st.markdown("## Review Scores Rating Prediction")
        with st.form("regression_form"):
            st.subheader("Property Details")
            col1, col2 = st.columns(2)
            with col1:
                property_type = st.selectbox("Property Type", ['Apartment', 'House', 'Bungalow', 'Guest suite', 'Condominium',
       'Cottage', 'Villa', 'Townhouse', 'Guesthouse', 'Other', 'Loft',
       'Resort', 'Casa particular (Cuba)', 'Boat', 'Hostel', 'Tiny house',
       'Bed and breakfast', 'Serviced apartment', 'Camper/RV',
       'Boutique hotel', 'Aparthotel', 'Campsite', 'Farm stay', 'Igloo',
       'Hotel', 'Cabin', 'Castle', 'Cave', 'Earth house', 'Treehouse',
       'Bus', 'Dome house', 'Tent', 'Chalet', 'Nature lodge',
       'Vacation home'])
                room_type = st.selectbox("Room Type", ['Private room', 'Entire home/apt', 'Shared room'])
                accommodates = st.number_input("Accommodates", min_value=1, max_value=16, value=2)
                beds = st.number_input("Beds", min_value=1, max_value=16, value=1)
                minimum_nights = st.number_input("Minimum Nights", min_value=1, value=2)
                maximum_nights = st.number_input("Maximum Nights", min_value=1, value=365)
                cleaning_fee = st.number_input("Cleaning Fee ($)", min_value=0, value=50)

            with col2:
                host_is_superhost = st.checkbox("Host is Superhost")
                instant_bookable = st.checkbox("Instant Bookable")
                is_location_exact = st.checkbox("Location Exact")
                host_total_listings_count = st.number_input("Host Total Listings", min_value=1, value=2)
                cancellation_policy = st.selectbox("Cancellation Policy", [
                    "flexible", "moderate", "strict", 
                    "strict_14_with_grace_period", "super_strict_30", "super_strict_60"
                ], index=1)
                guest_favorite = st.checkbox("Guest Favorite")
                host_rating = st.slider("Host Rating", 1.0, 5.0, 4.8, 0.1)

            st.subheader("Property Descriptions")
            summary = st.text_area("Summary", "Cozy apartment in central location")
            space = st.text_area("Space Description", "Comfortable living space with all amenities")
            description = st.text_area("Full Description", "This lovely apartment features...")
            notes = st.text_area("Notes", "Check-in instructions will be provided...")
            transit = st.text_area("Transit Information", "Easy access to public transportation...")
            access = st.text_area("Access Information", "Self check-in with keypad...")
            house_rules = st.text_area("House Rules", "No smoking, no parties...")
            neighborhood_overview = st.text_area("Neighborhood Overview", "Quiet residential area close to...")
            # interaction = st.text_area("Interaction Description", "I'm available if needed...")

            st.subheader("Amenities")
            amenities_dict = {
                "Essentials": ["Essentials", "Bath towel", "Bathroom essentials", "Bed linens", "Bedroom comforts",
        "Body soap", "Cooking basics", "Dishes and silverware", "Hangers", "Heating",
        "Hot water", "Internet", "Shampoo", "Toilet paper", "Wifi", "TV", "Cleaning before checkout",
        "Ethernet connection", "Hair dryer", "Hot water kettle", "toilet"],
                "Safety": ["Carbon monoxide detector", "Fire extinguisher", "First aid kit", "Safety card",
        "Smoke detector", "Window guards", "Buzzer/wireless intercom",
        "Lock on bedroom door", "Doorman", "Smart lock", "Keypad", "Fireplace guards"],
                "Luxury": [ "Air purifier", "Alfresco bathtub", "En suite bathroom", "Espresso machine", "Firm mattress",
        "Heated floors", "Heated towel rack", "Hot tub", "Jetted tub", "Memory foam mattress",
        "Pillow-top mattress", "Private hot tub", "Private pool", "Rain shower", "Sauna",
        "Soaking tub", "Sound system", "Stand alone steam shower", "Sun loungers", "Wine cooler",
        "Building staff", "Day bed", "Host greets you", "Indoor fireplace", "Luggage dropoff allowed",
        "Private bathroom", "Private entrance", "Private living room", "Room-darkening shades",
        "Suitable for events", "Ski-in/Ski-out", "Smoking allowed"],
                "Outdoor": [        "BBQ grill", "Balcony", "Beach essentials", "Beach view", "Beachfront",
        "Free parking on premises", "Free street parking", "Garden or backyard", "Hammock",
        "Lake access", "Mountain view", "Outdoor kitchen", "Outdoor parking",
        "Outdoor seating", "Patio or balcony", "Terrace", "Waterfront", "Tennis court",
        "Pool", "Pool toys", "Fire pit"],
                "Home Appliances": [        "Air conditioning", "Ceiling fan", "Central air conditioning", "Coffee maker",
        "Convection oven", "Dishwasher", "Dryer", "EV charger", "Exercise equipment",
        "Fax machine", "Full kitchen", "Gas oven", "Gym", "High-resolution computer monitor",
        "Kitchen", "Kitchenette", "Laptop friendly workspace", "Lockbox", "Long term stays allowed",
        "Microwave", "Mini fridge", "Murphy bed", "Oven", "Paid parking off premises",
        "Paid parking on premises", "Printer", "Refrigerator", "Stove", "Washer",
        "Warming drawer", "Pocket wifi", "Shared gym", "Shared hot tub", "Shared pool",
        "Self check-in", "Extra pillows and blankets", "Formal dining area", "Standing valet",
        "Iron", "Double oven", "Heat lamps", "Breakfast", "Breakfast table", "Bidet"]
            }

            selected_amenities = []
            for category, amenities in amenities_dict.items():
                with st.expander(category):
                    for amenity in amenities:
                        if st.checkbox(amenity, key=f"amenity_{amenity}"):
                            selected_amenities.append(amenity)

            submitted = st.form_submit_button("Predict Review Score")

            if submitted:
                input_data = {
                    "property_type": property_type.lower(),
                    "room_type": room_type.lower(),
                    "accommodates": accommodates,
                    "beds": beds,
                    "minimum_nights": minimum_nights,
                    "maximum_nights": maximum_nights,
                    "cleaning_fee": cleaning_fee,
                    "host_is_superhost": 1 if host_is_superhost else 0,
                    "instant_bookable": 1 if instant_bookable else 0,
                    "is_location_exact": 1 if is_location_exact else 0,
                    "host_total_listings_count": host_total_listings_count,
                    "cancellation_policy": cancellation_policy,
                    "guest_favorite": 1 if guest_favorite else 0,
                    "host_rating": host_rating,
                    "summary": summary,
                    "space": space,
                    "description": description,
                    "notes": notes,
                    "transit": transit,
                    "access": access,
                    "house_rules": house_rules,
                    "neighborhood_overview": neighborhood_overview,
                    # "interaction": interaction,
                    "amenities": ",".join(selected_amenities)
                }

                df = pd.DataFrame([input_data])

                @st.cache_data(show_spinner=False)
                def load_models():
                    with st.spinner('Loading models...'):
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

                        label_encoders = joblib.load('regression_models/label_encoders')
                        scaler = joblib.load('regression_models/scaler')
                        model = joblib.load('regression_models/stacked_model')

                        return (
                            summary_tfidf, space_tfidf, description_tfidf, notes_tfidf, transit_tfidf,
                            access_tfidf, house_rules_tfidf, svd_summary, svd_space, svd_transit,
                            svd_access, svd_description, svd_notes, svd_house_rules, label_encoders,
                            scaler, model
                        )

                (
                    summary_tfidf, space_tfidf, description_tfidf, notes_tfidf, transit_tfidf,
                    access_tfidf, house_rules_tfidf, svd_summary, svd_space, svd_transit,
                    svd_access, svd_description, svd_notes, svd_house_rules, label_encoders,
                    scaler, model
                ) = load_models()

                def apply_svd(df, column_name, tfidf_vectorizer, svd_model, prefix):
                    tfidf_matrix = tfidf_vectorizer.transform(df[column_name])
                    svd_features = svd_model.transform(tfidf_matrix)
                    for i in range(svd_features.shape[1]):
                        df[f"{prefix}_svd_{i}"] = svd_features[:, i]
                    return df

                df = apply_svd(df, "summary", summary_tfidf, svd_summary, "summary")
                df = apply_svd(df, "space", space_tfidf, svd_space, "space")
                df = apply_svd(df, "description", description_tfidf, svd_description, "description")
                df = apply_svd(df, "notes", notes_tfidf, svd_notes, "notes")
                df = apply_svd(df, "transit", transit_tfidf, svd_transit, "transit")
                df = apply_svd(df, "access", access_tfidf, svd_access, "access")
                df = apply_svd(df, "house_rules", house_rules_tfidf, svd_house_rules, "house_rules")

                for col in ['property_type', 'room_type']:
                    if col in label_encoders:
                        df[col] = label_encoders[col].transform(df[col])
                    else:
                        st.error(f"Missing label encoder for {col}")
                        st.stop()
                for category in ["Safety", "Outdoor", "Luxury", "Essentials", "Home Appliances"]:
                    df[category] = int(any(item in selected_amenities for item in amenities_dict[category]))

                selected_features = [
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

                missing = [col for col in selected_features if col not in df.columns]
                if missing:
                    st.error(f"Missing features in processed data: {missing}")
                    st.stop()

                cancellation_rank = {
                        'flexible': 5,
                        'moderate': 4,
                        'strict': 3,
                        'strict_14_with_grace_period': 2,
                        'super_strict_30': 1,
                        'super_strict_60': 0
                    }
                df['cancellation_policy'] = df['cancellation_policy'].map(cancellation_rank)


                sentiment_Cols = ['neighborhood_overview']

                for column_sent in sentiment_Cols:
                    df[column_sent] = df[column_sent].apply(lambda x: TextBlob(x).sentiment.polarity)

                X = df[selected_features]
                X_scaled = scaler.transform(X)
                y_pred = model.predict(X_scaled)

                st.success(f"Predicted Review Score: {y_pred[0]:.1f}/100")

                st.markdown("""
                **Interpretation:**
                - 90-100: Excellent (Top-rated listings)
                - 80-89: Very Good (Highly recommended)
                - 70-79: Good (Satisfactory experience)
                - Below 70: Needs improvement
                """)


    elif task_type == "Classification":
        st.markdown("## Guest Satisfaction Prediction")

        @st.cache_data(show_spinner=False)
        def load_models():
            with st.spinner('Loading models...'):
                transit_tfidf = joblib.load("classification_models/transit_tfidf_vectorizer")
                house_rules_tfidf = joblib.load("classification_models/house_rules_tfidf_vectorizer")
                svd_transit = joblib.load("classification_models/svd_transit_model")
                svd_house_rules = joblib.load("classification_models/svd_house_rules_model")
                label_encoders = joblib.load('classification_models/label_encoders')
                scaler = joblib.load('classification_models/scaler')
                model = joblib.load('classification_models/model')
                return (
                    transit_tfidf, house_rules_tfidf, svd_transit, svd_house_rules, label_encoders,
                    scaler, model
                )

        (
            transit_tfidf, house_rules_tfidf, svd_transit, svd_house_rules, label_encoders,
            scaler, model
        ) = load_models()

        with st.form("classification_form"):
            st.subheader("Property Details")
            col1, col2 = st.columns(2)
            with col1:
                property_type = st.selectbox("Property Type", label_encoders["property_type"].classes_)
                room_type = st.selectbox("Room Type", label_encoders["room_type"].classes_)
                host_is_superhost = st.checkbox("Host is Superhost")
                instant_bookable = st.checkbox("Instant Bookable")
                guest_favorite = st.checkbox("Guest Favorite")
                number_of_stays = st.number_input("Number of Stays", min_value=0, max_value=100, value=34, step=1)
                number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=100, value=17, step=1)

            with col2:
                host_rating = st.slider("Host Rating", 1.0, 5.0, 4.8, 0.1)
                host_total_listings_count = st.number_input("Host Total Listings", min_value=1, value=2)
                cancellation_policy = st.selectbox("Cancellation Policy", [
                    "flexible", "moderate", "strict",
                    "strict_14_with_grace_period", "super_strict_30", "super_strict_60"
                ], index=1)
                is_location_exact = st.checkbox("Location Exact")
                require_guest_phone_verification = st.checkbox("Require Phone Verification")

            st.subheader("Property Descriptions")
            transit = st.text_area("Transit Information", "Easy access to public transportation...")
            house_rules = st.text_area("House Rules", "No smoking, no parties...")

            st.subheader("Amenities")
            amenities_dict = {
                "Essentials": ["Essentials", "Bath towel", "Bathroom essentials", "Bed linens", "Bedroom comforts",
        "Body soap", "Cooking basics", "Dishes and silverware", "Hangers", "Heating",
        "Hot water", "Internet", "Shampoo", "Toilet paper", "Wifi", "TV", "Cleaning before checkout",
        "Ethernet connection", "Hair dryer", "Hot water kettle", "toilet"],
                "Safety": ["Carbon monoxide detector", "Fire extinguisher", "First aid kit", "Safety card",
        "Smoke detector", "Window guards", "Buzzer/wireless intercom",
        "Lock on bedroom door", "Doorman", "Smart lock", "Keypad", "Fireplace guards"],
                "Luxury": [ "Air purifier", "Alfresco bathtub", "En suite bathroom", "Espresso machine", "Firm mattress",
        "Heated floors", "Heated towel rack", "Hot tub", "Jetted tub", "Memory foam mattress",
        "Pillow-top mattress", "Private hot tub", "Private pool", "Rain shower", "Sauna",
        "Soaking tub", "Sound system", "Stand alone steam shower", "Sun loungers", "Wine cooler",
        "Building staff", "Day bed", "Host greets you", "Indoor fireplace", "Luggage dropoff allowed",
        "Private bathroom", "Private entrance", "Private living room", "Room-darkening shades",
        "Suitable for events", "Ski-in/Ski-out", "Smoking allowed"],
                "Outdoor": [        "BBQ grill", "Balcony", "Beach essentials", "Beach view", "Beachfront",
        "Free parking on premises", "Free street parking", "Garden or backyard", "Hammock",
        "Lake access", "Mountain view", "Outdoor kitchen", "Outdoor parking",
        "Outdoor seating", "Patio or balcony", "Terrace", "Waterfront", "Tennis court",
        "Pool", "Pool toys", "Fire pit"]
            }

            selected_amenities = []
            for category, amenities in amenities_dict.items():
                with st.expander(category):
                    for amenity in amenities:
                        if st.checkbox(amenity, key=f"amenity_{amenity}"):
                            selected_amenities.append(amenity)

            st.subheader("Additional Details")
            # col3, col4 = st.columns(2)
            # with col3:

             

            # with col4:
            neighbourhood_cleansed = st.selectbox("Neighbourhood Cleansed", label_encoders["neighbourhood_cleansed"].classes_)
            neighbourhood = st.selectbox("Neighbourhood", label_encoders["neighbourhood"].classes_)
            host_neighbourhood = st.selectbox("Host Neighbourhood", label_encoders["host_neighbourhood"].classes_)
            host_name = st.selectbox("Host Name", label_encoders["host_name"].classes_)

            submitted = st.form_submit_button("Predict Guest Satisfaction")

            if submitted:
                input_data = {
                    "property_type": property_type,
                    "room_type": room_type,
                    "host_is_superhost": 1 if host_is_superhost else 0,
                    "instant_bookable": 1 if instant_bookable else 0,
                    "is_location_exact": 1 if is_location_exact else 0,
                    "host_total_listings_count": host_total_listings_count,
                    "cancellation_policy": cancellation_policy,
                    "host_rating": host_rating,
                    "transit": transit,
                    "house_rules": house_rules,
                    "require_guest_phone_verification": 1 if require_guest_phone_verification else 0,
                    "amenities": ",".join(selected_amenities),
                    "number_of_stays": number_of_stays,
                    "number_of_reviews": number_of_reviews,
                    "neighbourhood_cleansed": neighbourhood_cleansed,
                    "neighbourhood": neighbourhood,
                    "host_neighbourhood": host_neighbourhood,
                    "host_name": host_name,
                    "guest_favorite": 1 if guest_favorite == "Yes" else 0
                }

                df = pd.DataFrame([input_data])

                def categorize_amenities(amenities_str):
                    from collections import defaultdict
                    amenities = amenities_str.split(",") if amenities_str else []
                    counts = defaultdict(int)
                    for amenity in amenities:
                        for category, items in amenities_dict.items():
                            if amenity in items:
                                counts[category] += 1
                    return counts

                amenity_counts = categorize_amenities(input_data["amenities"])
                for category in amenities_dict.keys():
                    df[category] = amenity_counts.get(category, 0)

                columns_to_encode = ['property_type', 'room_type', 'neighbourhood_cleansed',
                                    'neighbourhood', 'host_neighbourhood', 'host_name']
                for col in columns_to_encode:
                    if col in df.columns:
                        df[col] = label_encoders[col].transform(df[col])
                    else:
                        st.error(f"Missing label encoder for {col}")
                        st.stop()

                cancellation_rank = {
                    'flexible': 5,
                    'moderate': 4,
                    'strict': 3,
                    'strict_14_with_grace_period': 2,
                    'super_strict_30': 1,
                    'super_strict_60': 0
                }
                df['cancellation_policy'] = df['cancellation_policy'].map(cancellation_rank)

                # Text preprocessing
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                from nltk.tokenize import word_tokenize
                import nltk
                nltk.download('punkt')
                nltk.download('stopwords')
                nltk.download('wordnet')

                text_columns = ['transit', 'house_rules']
                stop_words = set(stopwords.words('english'))
                lemmatizer = WordNetLemmatizer()

                for col in text_columns:
                    df[col] = df[col].fillna("No information Provided").str.lower()
                    df[col] = df[col].str.replace(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", "", regex=True)
                    df[col] = df[col].str.replace(r"<.*?>", "", regex=True)
                    df[col] = df[col].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
                    df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
                    df[col] = df[col].apply(lambda x: ' '.join([
                        lemmatizer.lemmatize(word) for word in word_tokenize(x)
                        if word not in stop_words
                    ]))

                # Vectorize & reduce dimensionality
                transit_vec = transit_tfidf.transform(df['transit'])
                transit_svd = svd_transit.transform(transit_vec)
                df['transit_svd_1'] = transit_svd[:, 0]

                house_rules_vec = house_rules_tfidf.transform(df['house_rules'])
                house_rules_svd = svd_house_rules.transform(house_rules_vec)
                df['house_rules_svd_15'] = house_rules_svd[:, 0]

                selected_features = [
                    'host_rating', 'Luxury', 'Safety', 'property_type', 'number_of_stays',
                    'host_is_superhost', 'guest_favorite', 'is_location_exact', 'house_rules_svd_15',
                    'number_of_reviews', 'room_type', 'cancellation_policy', 'neighbourhood_cleansed',
                    'transit_svd_1', 'neighbourhood', 'Outdoor', 'host_name', 'host_neighbourhood',
                    'host_total_listings_count', 'Essentials', 'instant_bookable',
                    'require_guest_phone_verification'
                ]

                for feature in selected_features:
                    if feature not in df.columns:
                        df[feature] = 0

                X = df[selected_features]
                X_scaled = scaler.transform(X)

                y_pred = model.predict(X_scaled)

                guest_satisfaction_labels = {
                    2: 'Very High',
                    1: 'High',
                    0: 'Average'
                }
                predicted_label = guest_satisfaction_labels.get(y_pred[0], "Unknown")

                st.success(f"Predicted Guest Satisfaction: **{predicted_label}**")

                st.markdown("""
                **Interpretation:**
                - **Very High**: Exceptional experience, likely to receive top ratings and glowing reviews
                - **High**: Good experience with minor areas for improvement
                - **Average**: Needs significant improvements to meet guest expectations
                """)

elif page == "Metrics":
    task_type = st.sidebar.selectbox("Select Task Type", ["Regression", "Classification"])
    st.sidebar.markdown("View the performance metrics of the models.")

    if task_type == "Regression":
        st.markdown("## Regression Models Performance")

        data = {
            "Model": [
                "Linear Regression", "Lasso Regression", "Ridge Regression", "ElasticNet", "SVR",
                "Random Forest", "Decision Tree", "XGBoost", "CatBoost", "LightGBM", "Stacking"
            ],
            "R2 Score": [
                0.2589, 0.2586, 0.2587, 0.2575, 0.2353,
                0.2733, 0.1178, 0.2835, 0.2999, 0.2914, 0.3044
            ]
        }
        df = pd.DataFrame(data)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(df)

        with col2:
            bar_fig = go.Figure(go.Bar(
                x=df["R2 Score"],
                y=df["Model"],
                orientation='h',
                marker_color='#40E0D0'  # turquoise color
            ))
            bar_fig.update_layout(
                title="R² Scores for All Models",
                xaxis_title="R² Score",
                yaxis_title="Model",
                xaxis=dict(range=[0, 1]),
                margin=dict(l=150, r=20, t=30, b=20),
                height=400
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        best_row = df.loc[df["R2 Score"].idxmax()]
        best_model = best_row["Model"]
        best_r2 = best_row["R2 Score"]

        st.subheader(f"Best Model: {best_model} with R² Score = {best_r2:.4f}")

        colors = ['#E41D53', '#EE2D6B', '#F76EA0', '#F9AFC0']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best_r2,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "R² Score"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#40E0D0'},
                'steps': [
                    {'range': [0.0, 0.25], 'color': colors[0]},
                    {'range': [0.25, 0.5], 'color': colors[1]},
                    {'range': [0.5, 0.75], 'color': colors[2]},
                    {'range': [0.75, 1.0], 'color': colors[3]},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    elif task_type == "Classification":
        st.markdown("## Classification Models Performance")

        data = {
            "Model": [
                "Logistic Regression", "SVC", "Random Forest", "Decision Tree",
                "XGBoost", "CatBoost", "LightGBM", "Stacking", "Voting"
            ],
            "Accuracy Score": [
                0.5696, 0.5828, 0.6115, 0.5433,
                0.6258, 0.6332, 0.6309, 0.6470, 0.6338
            ]
        }
        df = pd.DataFrame(data)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.dataframe(df)

        with col2:
            bar_fig = go.Figure(go.Bar(
                x=df["Accuracy Score"],
                y=df["Model"],
                orientation='h',
                marker_color='#40E0D0'  # turquoise color
            ))
            bar_fig.update_layout(
                title="Accuracy Scores for All Models",
                xaxis_title="Accuracy Score",
                yaxis_title="Model",
                xaxis=dict(range=[0, 1]),
                margin=dict(l=150, r=20, t=30, b=20),
                height=400
            )
            st.plotly_chart(bar_fig, use_container_width=True)

        best_row = df.loc[df["Accuracy Score"].idxmax()]
        best_model = best_row["Model"]
        best_acc = best_row["Accuracy Score"]

        st.subheader(f"Best Model: {best_model} with Accuracy Score = {best_acc:.4f}")

        colors = ['#E41D53', '#EE2D6B', '#F76EA0', '#F9AFC0']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=best_acc,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Accuracy Score"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': '#40E0D0'},
                'steps': [
                    {'range': [0.0, 0.25], 'color': colors[0]},
                    {'range': [0.25, 0.5], 'color': colors[1]},
                    {'range': [0.5, 0.75], 'color': colors[2]},
                    {'range': [0.75, 1.0], 'color': colors[3]},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        
    
