from pydantic import BaseModel, validator, constr
import re

# User model
class User(BaseModel):
    username: str
    password: constr(min_length=8)  # Minimum length of 8 characters

    @validator('password')
    def validate_password(cls, password):
        # Check for at least one capital letter
        if not re.search(r'[A-Z]', password):
            raise ValueError('Password must contain at least one uppercase letter.')
        # Check for at least one number
        if not re.search(r'[0-9]', password):
            raise ValueError('Password must contain at least one number.')
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            raise ValueError('Password must contain at least one special character.')
        return password

class UserInDB(User):
    hashed_password: str

class SetupRequest(BaseModel):
    coldKey: str
    hotKey: str
    hotKeyMnemonic: str
    externalAPI: str
    minerId: int
    league_committed: str

class Token(BaseModel):
    access_token: str
    token_type: str