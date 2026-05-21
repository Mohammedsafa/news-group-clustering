# app.py

import uvicorn
from src.api.routes import app

if __name__ == "__main__":
    print("Launching Production Clustering Engine Server...")
    print("Access Swagger UI documentation at: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        "src.api.routes:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,       
        workers=1          
    )