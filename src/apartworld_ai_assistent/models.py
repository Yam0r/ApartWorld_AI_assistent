from datetime import date, datetime
from pydantic import BaseModel

class BookedPeriod(BaseModel):
    check_in: date        
    check_out: date       

class Unit(BaseModel):
    id: str              
    name: str
    district: str
    max_guests: int
    min_nights: int
    price_per_night_czk: int | None    
    calendar_updated_at: datetime
    booked: list[BookedPeriod] = []

class Listing(BaseModel):
    id: str              
    unit_id: str          
    channel: str          