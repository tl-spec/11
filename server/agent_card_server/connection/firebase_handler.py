from firebase_admin import credentials
from firebase_admin import db
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials
import firebase_admin
from dotenv import load_dotenv
import pickle
import uuid 
import os
import uuid


if os.path.exists(".env.connection") or os.path.exists("server/.env.connection"):
    load_dotenv(".env.connection") 
    load_dotenv("server/.env.connection")
elif os.path.exists(".env.connection.docker") or os.path.exists("server/.env.connection.docker"):
    load_dotenv(".env.connection.docker")
    load_dotenv("server/.env.connection.docker")
else:
    print("Error: .env.connection or .env.connection.docker file not found. Please create a .env.connection or .env.connection.docker file in the agent_card_server/connection directory.")
    exit(1)


firebaseApp = None 

def initialize_firebase():
    global firebaseApp  
    cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS"))
    app = firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv("FIREBASE_DATABASE_URL"),
        'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
    })
    firebaseApp = app
    return app 

def get_db():
    if firebaseApp is None:
        initialize_firebase()
    return db


def get_storage():
    if firebaseApp is None:
        initialize_firebase()
    client = storage.Client(
        credentials=AnonymousCredentials(),
        project="insightagent-local"
    )
    return client


def download_file(source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    bucket = get_storage().bucket("local-storage-bucket")
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
    return destination_file_name


def download_projects_from_firebase(projectUid):
    """Downloads all projects from the bucket."""
    if os.path.isdir(f"./agent_card_server/data/projects/{projectUid}"):
        print(f"Projects already downloaded.")
        return f"./agent_card_server/data/projects/{projectUid}"
    bucket = get_storage().bucket("local-storage-bucket")
    blobs = bucket.list_blobs(prefix=f"projects/{projectUid}")
    for blob in blobs:
        if not os.path.isdir(f"./agent_card_server/data/projects/{projectUid}"):
            os.mkdir(f"./agent_card_server/data/projects/{projectUid}")
        blob.download_to_filename(f"./agent_card_server/data/{blob.name}")
    print(f"All blobs downloaded to projects folder.")
    return f"./agent_card_server/data/projects/{projectUid}"


def list_all_projects(): 
    """Get all projects from db.reference("projects/{projectUid}")"""
    db = get_db() 
    ref = db.reference("projects")
    return ref.get()

    
def save_project(projectId, projectInfo, local_path):
    db = get_db() #数据库连接
    ref = db.reference(f"projects/{projectId}")#将projectInfo写入到路径为projects/{projectId}的数据库引用中。
    ref.set(projectInfo)#保存projectInfo
    
    bucket = get_storage().bucket("local-storage-bucket")
    #使用get_storage()获取存储客户端，从本地路径上传两个索引文件（index.faiss和index.pkl）到指定的云存储桶中对应的项目目录下
    indexBlob = bucket.blob(f"projects/{projectId}/index.faiss")#创建一个指向云存储桶中特定位置的blob对象
    indexBlob.upload_from_filename(f"{local_path}/index.faiss")#将本地文件上传到云存储位置
    
    indexPKLBlob = bucket.blob(f"projects/{projectId}/index.pkl")#创建一个指向云存储桶中特定位置的blob对象
    indexPKLBlob.upload_from_filename(f"{local_path}/index.pkl", timeout=30000)
    #将本地文件上传到云存储位置，设置超时时间为30000毫秒
    
  
    
def save_available_computations(projectId: str, queryName: str, computationId: str, data):
    db = get_db() 
    ref = db.reference(f"projects/{projectId}/computations")
    ref.update({
        computationId: {
            "computationId": computationId,
            "queryName": queryName,
        }, 
    })
    ## save it to bucket as well
    filename = f"./agent_card_server/data/computations/{computationId}.pkl"
    with open(filename, "wb") as f: 
        pickle.dump(data, f)
    bucket = get_storage().bucket("local-storage-bucket")
    blob = bucket.blob(f"computations/{computationId}.pkl")
    blob.upload_from_filename(filename)
    
def load_computations_pickle(computationId: str):
    filename = f"./agent_card_server/data/computations/{computationId}.pkl"
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        bucket = get_storage().bucket("local-storage-bucket")
        blob = bucket.blob(f"computations/{computationId}.pkl")
        blob.download_to_filename(filename)

    with open(filename, "rb") as f: 
        return pickle.load(f)
    

def get_available_computations_for_env(projectId: str):
    db = get_db() 
    ref = db.reference(f"projects/{projectId}/computations")
    return ref.get()

def create_user_study_section(user_study_id: str, uid: str, projectId: str=None, ): 
    db = get_db()
    ref = db.reference(f"user_studies/{user_study_id}")
    ref.set({
        "user_study_id": user_study_id,
        "uid": uid,
        'projectId': projectId if projectId is not None else None,
    })
    print(f"User study section created for {user_study_id} with {uid} and {projectId}")
    
def update_user_study_section(user_study_id: str, uid: str, projectId: str=None, ):
    db = get_db()
    ref = db.reference(f"user_studies/{user_study_id}")
    ref.update({
        "user_study_id": user_study_id, 
        "uid": uid, 
        'projectId': projectId if projectId is not None else None,
    })
    print(f"User study section updated for {user_study_id} with {uid} and {projectId}")

    
def update_agent_creation_in_study(user_study_id: str, agent_id: str, agent_type: str, agent_name: str, agent_config: any):
    db = get_db()
    ref = db.reference(f"user_studies/{user_study_id}/agents/{agent_id}")
    ref.update({
        "agent_id": agent_id,
        "agent_type": agent_type,
        "agent_name": agent_name,
        "agent_config": agent_config,
    })
    
def update_agent_in_study(user_study_id: str, agent_id: str, agent_type: str, agent_name: str, agent_config: any):
    db = get_db()
    ref = db.reference(f"user_studies/{user_study_id}/agents/{agent_id}")
    ref.update({
        "agent_id": agent_id,
        "agent_type": agent_type,
        "agent_name": agent_name,
        "agent_config": agent_config,
    })
    
def create_branch(user_study_id: str, agent_id: str, branchId: str, timestamp: str):
    db = get_db()
    print(f"Creating branch {branchId} for agent {agent_id} in user study {user_study_id}")
    ref = db.reference(f"user_studies/{user_study_id}/agents/{agent_id}/branches/{branchId}")
    ref.update({
        "branch_id": branchId,
        "timestamp": timestamp,
    })
   

def save_commit(user_study_id: str, agent_id, branchId: str, commit: any):
    db = get_db()
    ref = db.reference(f"user_studies/{user_study_id}/agents/{agent_id}/branches/{branchId}/commits")
    ref.update({
        f"commit-{commit.index_in_branch}{'(interact)' if commit.human_interaction else '('+commit.blobs[0].task_type+')'}": commit.to_dict()
    })


# def load_computations_pickle(computationId: str):
#     filename = f"./agent_card_server/data/computations/{computationId}.pkl"
#     # check if file exists 
#     if not os.path.exists(filename):
#         print(f"File {filename} not found.")
#         bucket = get_storage().bucket()
#         # download the file
#         blob = bucket.blob(f"computations/{computationId}.pkl")
#         blob.download_to_filename(filename)

#     with open(filename, "rb") as f: 
#         return pickle.load(f)    

def download_ris(filePath: str):
    bucket = get_storage().bucket("local-storage-bucket")#调用get_storage()方法获取Firebase存储客户端
    blob = bucket.blob(filePath) #定位到“local-storage-bucket"存储桶
    #根据传入文件路径获取对应的blob对象
    download_to_filename = f"./agent_card_server/data/raw/{uuid.uuid4()}.ris"#生成本地保存路径（使用UUID确保文件名唯一）
    blob.download_to_filename(download_to_filename)#将文件下载到指定位置并返回保存的文件路径
    return download_to_filename
    
    


    
    