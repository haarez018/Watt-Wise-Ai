from pydantic import BaseModel, EmailStr, Field


class OAuthExchangeRequest(BaseModel):
    email: EmailStr
    full_name: str | None = None
    provider: str = Field(min_length=1, max_length=50)
    provider_subject: str = Field(min_length=1, max_length=255)
