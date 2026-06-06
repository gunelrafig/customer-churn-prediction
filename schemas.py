from pydantic import BaseModel, Field

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=72)
    monthly_charges: float = Field(..., ge=0, le=150)
    total_charges: float = Field(..., ge=0, le=10000)
    contract: str
    internet_service: str
    payment_method: str
    gender: str
    partner: str
    dependents: str