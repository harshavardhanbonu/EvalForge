from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Pydantic will automatically read these variables from your .env file.
    If they are missing, the server will crash on startup, protecting you 
    from running the app without a database or API key.
    """
    DATABASE_URL: str
    GEMINI_API_KEY: str

    class Config:
        env_file = ".env"

settings = Settings()