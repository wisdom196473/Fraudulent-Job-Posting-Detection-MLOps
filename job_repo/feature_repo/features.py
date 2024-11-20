from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, Project
from feast.types import Float32, String, Int64

# Define a project for the feature repo
project = Project(name="job_project", description="A project for job data")

# Define the entity
job = Entity(
    name="job_id",
    join_keys=["job_id"],
    description="Unique identifier for job",
)

# Define the Structured Feature data source
job_structured_data_source = FileSource(
    name='job_structured_data',
    path="data/structured_features.parquet",
    timestamp_field="event_timestamp")

# Define the Structured Feature
job_structured_features = FeatureView(
    name="job_structured_features_view",
    entities=[job],
    ttl=timedelta(days=1),
    schema=[
        Field(name="location", dtype=Int64),
        Field(name="employment_type", dtype=Int64),
        Field(name="required_experience", dtype=Int64),
        Field(name="required_education", dtype=Int64),
        Field(name="industry", dtype=Int64),
        Field(name="function", dtype=Int64),
        Field(name="telecommuting", dtype=Int64),
        Field(name="has_company_logo", dtype=Int64),
        Field(name="has_questions", dtype=Int64),
        Field(name="no_logo_no_questions", dtype=Int64),
        Field(name="location_freq", dtype=Float32),
        Field(name="employment_type_freq", dtype=Float32),
        Field(name="required_experience_freq", dtype=Float32),
        Field(name="required_education_freq", dtype=Float32),
        Field(name="industry_freq", dtype=Float32),
        Field(name="function_freq", dtype=Float32),
        Field(name="description_is_missing", dtype=Int64),
        Field(name="requirements_is_missing", dtype=Int64),
        Field(name="company_profile_is_missing", dtype=Int64),
        Field(name="benefits_is_missing", dtype=Int64),
        Field(name="location_is_missing", dtype=Int64),
        Field(name="employment_type_is_missing", dtype=Int64),
        Field(name="required_experience_is_missing", dtype=Int64),
        Field(name="required_education_is_missing", dtype=Int64),
        Field(name="industry_is_missing", dtype=Int64),
        Field(name="function_is_missing", dtype=Int64),
        Field(name="title_length", dtype=Int64),
        Field(name="description_length", dtype=Int64),
        Field(name="requirements_length", dtype=Int64),
        Field(name="company_profile_length", dtype=Int64),
        Field(name="benefits_length", dtype=Int64),
    ],
    online=True,
    source=job_structured_data_source,
)

# Define the Text Features data source
job_text_data_source  = FileSource(
    name='job_text_data',
    path="data/text_features.parquet",
    timestamp_field="event_timestamp" )

# Define the predictor Feature View
job_text_features = FeatureView(
    name="job_text_features_view",
    entities=[job], 
    ttl=timedelta(days=1),
    schema=[
        Field(name="title_cleaned", dtype=String),
        Field(name="description_cleaned", dtype=String),
        Field(name="requirements_cleaned", dtype=String),
        Field(name="company_profile_cleaned", dtype=String),
        Field(name="benefits_cleaned", dtype=String)
    ],
    online=True,
    source=job_text_data_source,
)

# Define the target data source
job_target_source = FileSource(
    name='job_target',
    path="data/target_features.parquet",
    timestamp_field="event_timestamp")

# Define the target Feature View with all columns
job_target_features = FeatureView(
    name="job_target_feature_view",
    entities=[job],
    ttl=timedelta(days=1),
    schema=[
        Field(name="fraudulent", dtype=Int64),
    ],
    online=True,
    source=job_target_source,
)