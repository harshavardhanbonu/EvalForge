from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

# We use SQLAlchemy for our ORM. 
# Notice we rely on the DATABASE_URL from our config.py
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True, # Verifies connection is alive before using it
    pool_size=5,        # Keeps connection pool small for local dev
    max_overflow=10
)

# SessionLocal will be used to spawn individual database sessions per API request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# All our future database models will inherit from this Base
Base = declarative_base()

# Dependency function to use in FastAPI routes
def get_db():
    """
    Yields a database session and safely closes it after the request completes.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()