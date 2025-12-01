from firebase_functions import https_fn, options
from firebase_admin import db
from module.utils.helpers import authenticate_auth
from module.utils.user import create_default_user_profile 
import copy
import uuid
@https_fn.on_call()
def on_request_example(req: https_fn.CallableRequest) -> dict:
    uid = req.auth.uid
    return {'data': f"Hello {uid}! new"}


@https_fn.on_call()
def load_user_profile(req: https_fn.CallableRequest): 
    authenticate_auth(req)
    uid = req.auth.uid
    name = req.auth.token.get("name") or "Unknown"
    email = req.auth.token.get("email") or "Unknown"
    userSnapShot = db.reference(f"/users/{uid}").get()
    if userSnapShot is None:
        user = create_default_user_profile(uid, name, email)
        print(f"create a new user profile for uid: {uid}")
        db.reference(f"/users/{uid}").set(user)
        return user
    else: 
        return userSnapShot