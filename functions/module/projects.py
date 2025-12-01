from firebase_functions import https_fn, options
from firebase_admin import db
from module.utils.helpers import authenticate_auth

@https_fn.on_call()
def load_existing_user_study_envs(req: https_fn.CallableRequest): 
    authenticate_auth(req)
    uid = req.auth.uid
    name = req.auth.token.get("name") or "Unknown"
    email = req.auth.token.get("email") or "Unknown"
    envSnapShot = db.reference(f"/projects").get()
    if envSnapShot is None:
        return []
    else: 
        return envSnapShot