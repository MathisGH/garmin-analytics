from src.auth import get_authenticated_client
from src.database import create_schema, DB_PATH
from src.extract import extract_day
from datetime import date

client = get_authenticated_client()
print("Connection successful -> Name retrieved:", client.get_full_name())

create_schema(DB_PATH)

today = date.today().isoformat()
extract_day(client, DB_PATH, today)