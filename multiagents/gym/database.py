import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional

class Database:
    def __init__(self, db_path: str = "gym.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    profile TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Workouts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workouts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    plan TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            # Progress table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    workout_id INTEGER,
                    completion_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (workout_id) REFERENCES workouts (id)
                )
            """)
            
            # Conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    agent_name TEXT,
                    message TEXT,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """)
            
            conn.commit()
    
    def save_user_profile(self, user_id: str, profile: Dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO users (id, profile, updated_at)
                VALUES (?, ?, ?)
            """, (user_id, json.dumps(profile), datetime.now()))
            conn.commit()
    
    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT profile FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            return json.loads(result[0]) if result else None
    
    def save_workout_plan(self, user_id: str, plan: Dict) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO workouts (user_id, plan)
                VALUES (?, ?)
            """, (user_id, json.dumps(plan)))
            conn.commit()
            return cursor.lastrowid or 0
    
    def save_progress(self, user_id: str, workout_id: int, completion_data: Dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO progress (user_id, workout_id, completion_data)
                VALUES (?, ?, ?)
            """, (user_id, workout_id, json.dumps(completion_data)))
            conn.commit()
    
    def save_conversation(self, user_id: str, agent_name: str, message: str, response: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO conversations (user_id, agent_name, message, response)
                VALUES (?, ?, ?, ?)
            """, (user_id, agent_name, message, response))
            conn.commit() 