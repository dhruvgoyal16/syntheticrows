from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import pandas as pd
import io

app = FastAPI(
    title="SynthIQ API",
    description="Synthetic data generation engine",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "SynthIQ backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    summary = {
        "filename": file.filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "preview": df.head(5).to_dict(orient="records")
    }

    return summary

@app.post("/generate")
async def generate_synthetic_data(
    file: UploadFile = File(...),
    num_rows: int = 500
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Read uploaded CSV
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Limit free tier to 500 rows output
    if num_rows > 500:
        num_rows = 500

    try:
        # Detect metadata automatically
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # Train CTGAN
        synthesizer = CTGANSynthesizer(metadata, epochs=100, verbose=False)
        synthesizer.fit(df)

        # Generate synthetic rows
        synthetic_df = synthesizer.sample(num_rows=num_rows)

        # Return as downloadable CSV
        output = io.StringIO()
        synthetic_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_{file.filename}"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))