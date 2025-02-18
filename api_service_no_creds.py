import os
#from database import SessionLocal, Base
from fastapi import Depends, FastAPI, Response, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from schema import PostGet
from typing import List
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from sqlalchemy import TIMESTAMP, Column, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from hashlib import md5
from random import randint, seed

seed(42)

def get_exp_group(user_id: int) -> str:
    salt = randint(0, 1000)
    salted_id = str(user_id) + str(salt)
    group = int(md5(salted_id.encode()).digest()[-1]) % 2
    if group:
        return 'test'
    return 'control'

app = FastAPI()

SQLALCHEMY_DATABASE_URL = DB_URL

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class PostGet(BaseModel):
    #__tablename__ = "post_text_df"
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

class Post(Base):
    __tablename__ = "post_text_df"
    post_id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

class User(Base):
    __tablename__ = "user_data"
    age = Column(Integer)
    city = Column(String)
    country = Column(String)
    exp_group = Column(Integer)
    gender = Column(Integer)
    user_id = Column(Integer, primary_key=True)
    os = Column(String)
    source = Column(String)

class Feed(Base):
    __tablename__ = "feed_data"
    id = Column(Integer, primary_key=True)
    post_id = Column(Integer, ForeignKey("post_text_df.post_id"))
    post = relationship("Post")
    user_id = Column(Integer, ForeignKey("user_data.user_id"))
    user = relationship("User")
    action = Column(String)
    time = Column(TIMESTAMP)

class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]

if __name__ == '__main__':
    Base.metadata.create_all()

def get_db():
    with SessionLocal() as db:
        return db

def get_model_path(path: str, type: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model_' + type
    else:
        MODEL_PATH = path
    return MODEL_PATH

seed(42)
salt_dict = {}

def get_exp_group(user_id: int) -> str:
    if user_id in salt_dict:
        salt = salt_dict[user_id]
    else:
        salt = randint(0, 1000)
        salt_dict[user_id] = salt
    
    salted_id = str(user_id) + str(salt)
    group = int(md5(salted_id.encode()).digest()[-1]) % 2
    if group:
        return 'test'
    return 'control'

def load_model_control():
    model_path = get_model_path("catboost_model_5", "control")
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file

def load_models_test():
    model_path = get_model_path("catboost_model_v4", "test")
    from_file = CatBoostClassifier()
    from_file.load_model(model_path)
    return from_file

model_control = load_model_control()
model_test = load_models_test()

cat_features = ['gender', 'country', 'city', 'exp_group', 'os', 'source', 'topic']

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        POSTGRES_TOKEN
        POSTGRES_PORT
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_features() -> pd.DataFrame:
    query = f"SELECT * FROM IV_TABLE;"
    return batch_load_sql(query)

def load_posts() -> pd.DataFrame:
    query = f"SELECT * FROM public.post_text_df;"
    return batch_load_sql(query)

feat = load_features()
posts = load_posts()

@app.get("/post/recommendations/", response_model=Response)
def get_recommendations(id: int, time: datetime, limit: int = 5, db: Session = Depends(get_db)):
    exp_group = get_exp_group(id)
    if exp_group == 'control':
        model = model_control
    elif exp_group == 'test':
        model = model_test
    else:
        raise ValueError('unknown group')
    features_df = feat[feat['user_id'] == id]
    preds = pd.DataFrame(columns=['post_id', 'probability'])
    i = -1
    for row in features_df.values:
        i += 1
        post, features = row[2], row[3:]
        single_pred = model.predict_proba(features)
        preds.loc[i] = [int(post), single_pred[1]]
    preds.sort_values('probability', ascending=False, inplace=True)
    list_of_posts = [int(x) for x in preds['post_id'].loc[:limit]]

    post_objects = []
    for post_num in list_of_posts:
        post_to_recommend = posts[posts['post_id'] == post_num]
        post_object = PostGet(id=post_to_recommend['post_id'], text=str(post_to_recommend['text']), topic=str(post_to_recommend['topic']))
        post_objects.append(post_object)

    return Response(exp_group=exp_group, recommendations=post_objects)