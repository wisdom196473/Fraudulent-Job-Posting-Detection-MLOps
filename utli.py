import string
import os
import re
import pandas as pd
import contractions
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import AutoTokenizer
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from pandarallel import pandarallel
pandarallel.initialize()

stop_words = set(stopwords.words('english'))  # Set of English stopwords
lemmatizer = WordNetLemmatizer()
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm") #python -m spacy download en_core_web_sm

# Select the Features with Domain Knowledge

def select_features(df):
    """Select relevant features from the dataset"""
    selected_features = ['title', 'description', 'requirements', 'company_profile',
                         'benefits', 'location', 'employment_type', 'required_experience',
                         'required_education', 'industry', 'function', 'telecommuting',
                         'has_company_logo', 'has_questions', 'fraudulent']

    return df[selected_features]


# Handling Missing Values

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Create missing value indicators
    missing_indicators = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_indicators[f'{col}_is_missing'] = df[col].isnull().astype(int)

    # Handle text columns
    text_columns = ['title', 'description', 'requirements', 'company_profile', 'benefits']
    for col in text_columns:
        df.loc[:, col] = df[col].fillna('')

    # Handle categorical columns
    categorical_mappings = {
        'location': 'Unknown Location',
        'employment_type': 'Not Specified',
        'required_experience': 'Not Specified',
        'required_education': 'Not Specified',
        'industry': 'Unknown Industry',
        'function': 'Unknown Function'
    }

    for col, placeholder in categorical_mappings.items():
        df.loc[:, col] = df[col].fillna(placeholder)

    # Add missing indicators to the dataframe
    for col_name, indicator in missing_indicators.items():
        df.loc[:, col_name] = indicator

    return df




def clean_text(text):
    """Clean, tokenize, and remove stopwords from text data."""

    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)
    doc = nlp(cleaned_text)
    normalized_words = [token.lemma_ for token in doc]
    normalized_text = ' '.join(normalized_words)

    return normalized_text


# Prepare the First Feature

def prepare_initial_features(df):
    """Prepare initial features including text cleaning using Pandarallel"""
    df = handle_missing_values(df)

    text_columns = ['title', 'description', 'requirements', 'company_profile', 'benefits']
    for col in text_columns:
        print(f"Processing {col}...")
        df[f'{col}_cleaned'] = df[col].parallel_apply(clean_text)

    df['no_logo_no_questions'] = ((df['has_company_logo'] == 0) &
                                  (df['has_questions'] == 0)).astype(int)

    return df


def create_structured_features(df_processed):
    """Create structured feature store"""
    structured_features = pd.DataFrame()

    # 1. Label encode categorical features
    categorical_columns = ['location', 'employment_type', 'required_experience',
                           'required_education', 'industry', 'function']

    label_encoders = {}
    for col in categorical_columns:
        print(f"Label encoding {col}...")
        le = LabelEncoder()
        structured_features[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le

    # 2. Add binary features
    binary_features = ['telecommuting', 'has_company_logo', 'has_questions',
                       'no_logo_no_questions']
    structured_features[binary_features] = df_processed[binary_features]

    # 3. Add frequency encoded features
    for col in categorical_columns:
        freq_encoding = df_processed[col].value_counts(normalize=True)
        structured_features[f'{col}_freq'] = df_processed[col].map(freq_encoding)

    # 4. Add missing indicators
    missing_indicators = [col for col in df_processed.columns if col.endswith('_is_missing')]
    structured_features[missing_indicators] = df_processed[missing_indicators]

    # 5. Add text length features
    text_columns = ['title', 'description', 'requirements', 'company_profile', 'benefits']
    for col in text_columns:
        structured_features[f'{col}_length'] = df_processed[f'{col}_cleaned'].str.len()

    return structured_features, label_encoders


# Prepare the Second Feature Store

def create_text_features(df_processed):
    """Create text feature store"""
    text_features = pd.DataFrame()
    text_columns = ['title', 'description', 'requirements', 'company_profile', 'benefits']

    for col in text_columns:
        text_features[f'{col}_cleaned'] = df_processed[f'{col}_cleaned']

    return text_features


def print_feature_summary(structured_features, text_features):
    """Print summary of created features"""
    print("\nFeature Stores Summary:")
    print(f"Structured Features Shape: {structured_features.shape}")
    print(f"Number of structured features: {len(structured_features.columns)}")

    # Group features by type
    categorical_features = ['location', 'employment_type', 'required_experience',
                            'required_education', 'industry', 'function']
    binary_features = ['telecommuting', 'has_company_logo', 'has_questions', 'no_logo_no_questions']
    frequency_features = [col for col in structured_features.columns if col.endswith('_freq')]
    length_features = [col for col in structured_features.columns if col.endswith('_length')]

    print("\nStructured features include:")
    print(f"- Label encoded categorical features: {len(categorical_features)}")
    print(f"- Binary features: {len(binary_features)}")
    print(f"- Frequency encoded features: {len(frequency_features)}")
    print(f"- Text length features: {len(length_features)}")

    print(f"\nText Features Shape: {text_features.shape}")
    print("Text features include:", text_features.columns.tolist())


def save_features(structured_features, text_features, target_features, path='./job_repo/feature_repo/data/'):
    """Save feature stores to disk in Parquet format."""

    # Ensure the directory exists; create it if not
    os.makedirs(path, exist_ok=True)

    # Save structured features as Parquet
    structured_features.to_parquet(os.path.join(path, 'structured_features.parquet'), index=False)

    # Save text features as Parquet
    text_features.to_parquet(os.path.join(path, 'text_features.parquet'), index=False)

    # Save target features as Parquet
    target_features.to_parquet(os.path.join(path, 'target_features.parquet'), index=False)
