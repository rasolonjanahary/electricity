import joblib
import uvicorn
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Charger le modèle
model = joblib.load("model/regmodel.pkl")

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    year: int = Form(...),
    month: int = Form(...),
    stateDescription: str = Form(...),
    sectorName: str = Form(...),
    customers: float = Form(...)
):
    # Préparer les données
    input_data = pd.DataFrame([{
        "year": year,
        "month": month,
        "stateDescription": stateDescription,
        "sectorName": sectorName,
        "customers": customers
    }])

    # Assurez-vous que les colonnes correspondent au modèle
    prediction = model.predict(input_data)[0]
    return {"predicted_sales": prediction}

if __name__ == "__main__":
    uvicorn.run(app)
