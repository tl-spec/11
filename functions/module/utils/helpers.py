from firebase_functions import https_fn 
def authenticate_auth(context: https_fn.CallableRequest): 
    if context.auth is None: 
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.FAILED_PRECONDITION,
            message="The function must be called while authenticated.",
        )
