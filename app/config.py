import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_cloud_project: str
    google_cloud_location: str

    mongodb_uri: str
    mongodb_db: str
    mongodb_collection: str
    mongodb_vector_index: str

    gcs_bucket_name: str | None = None


def get_settings() -> Settings:
    required_vars = [
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "MONGODB_URI",
        "MONGODB_DB",
        "MONGODB_COLLECTION",
        "MONGODB_VECTOR_INDEX",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

    return Settings(
        google_cloud_project=os.environ["GOOGLE_CLOUD_PROJECT"],
        google_cloud_location=os.environ["GOOGLE_CLOUD_LOCATION"],
        mongodb_uri=os.environ["MONGODB_URI"],
        mongodb_db=os.environ["MONGODB_DB"],
        mongodb_collection=os.environ["MONGODB_COLLECTION"],
        mongodb_vector_index=os.environ["MONGODB_VECTOR_INDEX"],
        gcs_bucket_name=os.getenv("GCS_BUCKET_NAME"),
    )