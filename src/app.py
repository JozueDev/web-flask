from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

modelo = joblib.load("modelo_clima.pkl")
columnas = joblib.load("columnas_modelo.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        datos = pd.DataFrame([{
            "precipitation": float(request.form["precipitation"]),
            "temp_max": float(request.form["temp_max"]),
            "temp_min": float(request.form["temp_min"]),
            "wind": float(request.form["wind"])
        }])
        resultado = modelo.predict(datos)[0]

    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)



