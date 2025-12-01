def create_default_user_profile(uid, name=None, email=None): 
    """
    """
    user = {
        "uid": uid,
        "email": email or "Unknown", 
        "name": name or "Unknown",
        "can_participate_in_studies": True,
        "projects": [],
        "studies": [],  
    }
    return user