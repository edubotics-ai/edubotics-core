# TODO: Seperate into modules, maybe in edubotics-core?

from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import json
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional


class Database(ABC):
    @abstractmethod
    def create_session(self):
        """Create a new session or connection."""
        pass

    @abstractmethod
    def add(self, session: Any, data: dict):
        """Add a new record to the database."""
        pass

    @abstractmethod
    def get(self, session: Any, session_token: str) -> Optional[dict]:
        """Retrieve a record from the database by session token."""
        pass

    @abstractmethod
    def delete(self, session: Any, session_token: str):
        """Delete a record from the database by session token."""
        pass

    @abstractmethod
    def commit(self, session: Any):
        """Commit the current transaction."""
        pass

    @abstractmethod
    def close(self, session: Any):
        """Close the session or connection."""
        pass


# SQLite Database Implementation
DATABASE_URL = "sqlite:///:memory:"

# Create the database engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class for declarative class definitions
Base = declarative_base()


class SessionModel(Base):
    __tablename__ = "sessions"

    session_token = Column(String, primary_key=True, index=True)
    email = Column(String, index=True)
    name = Column(String)
    profile_image = Column(String)
    google_signed_in = Column(Boolean, default=False)
    literalai_info = Column(String)  # This will store the JSON string
    last_login = Column(DateTime)


# Create the database tables
Base.metadata.create_all(bind=engine)


class SQLiteDatabase(Database):
    def __init__(self, db_url="sqlite:///sessions.db"):
        self.engine = create_engine(
            db_url, poolclass=QueuePool, pool_size=20, max_overflow=0
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        self.session_cache = {}
        self.cache_lock = threading.Lock()

    def create_session(self):
        return self.SessionLocal()

    def add(self, session, data: dict):
        session_data = SessionModel(**data)
        session.add(session_data)

    def get(self, session, session_token: str) -> Optional[dict]:
        with self.cache_lock:
            cached_data = self.session_cache.get(session_token)
        if cached_data:
            return cached_data

        session_data = (
            session.query(SessionModel)
            .filter(SessionModel.session_token == session_token)
            .first()
        )
        if session_data:
            result = self._create_session_result_with_dynamic_keys(
                session_data, literalai_info=json.loads(session_data.literalai_info)
            )
            with self.cache_lock:
                self.session_cache[session_token] = result
            return result
        return None

    def delete(self, session, session_token: str):
        session.query(SessionModel).filter(
            SessionModel.session_token == session_token
        ).delete()

    def commit(self, session):
        session.commit()

    def close(self, session):
        session.close()

    def _create_session_result_with_dynamic_keys(self, session_data, **kwargs):
        result = {
            "session_token": session_data.session_token,
            "email": session_data.email,
            "name": session_data.name,
            "profile_image": session_data.profile_image,
            "google_signed_in": session_data.google_signed_in,
            "last_login": session_data.last_login,
        }
        result.update(kwargs)  # Dynamically add additional key-value pairs
        return result


class InMemoryDatabase(Database):
    def __init__(self):
        self.data_store = {}

    def create_session(self):
        # No actual session needed for in-memory dictionary
        return self.data_store

    def add(self, session, data: dict):
        session_token = data.get("session_token")
        if session_token:
            session[session_token] = data

    def get(self, session, session_token: str) -> Optional[dict]:
        return session.get(session_token)

    def delete(self, session, session_token: str):
        if session_token in session:
            del session[session_token]

    def commit(self, session):
        # No commit needed for in-memory dictionary
        pass

    def close(self, session):
        # No close operation needed for in-memory dictionary
        pass


class DatabaseFactory:
    @staticmethod
    def create_database(database_type: str) -> Database:
        if database_type == "sqlite":
            return SQLiteDatabase()
        elif database_type == "in_memory_dict":
            return InMemoryDatabase()
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
