# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_admin import initialize_app
from module.account import on_request_example, load_user_profile
from module.projects import load_existing_user_study_envs

initialize_app()
