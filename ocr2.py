import os
import json
import base64
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from groq import Groq
import warnings
import logging

# Silence Google GenAI SDK warnings
warnings.filterwarnings("ignore", message="Direct use of automatic function calling")
logging.getLogger("google_genai").setLevel(logging.ERROR)

# 1. Define the extraction schema
class LandRecord(BaseModel):
    state: str = Field(description="State name")
    district: str = Field(description="District name")
    taluka_or_tehsil: str = Field(description="Taluka / Tehsil / Block")
    village: str = Field(description="Village name")
    survey_or_gat_no: str = Field(description="Survey / Khasra / Gat number")
    khata_no: str = Field(description="Khata / Account number")
    total_area: str = Field(description="Total area with measurement units (e.g., 1.45 Hectare, Bigha, Acre)")
    owners: list[str] = Field(description="List of all owner names in the original script")
    liabilities_or_loans: list[str] = Field(default=[], description="Active bank loans, encumbrances, or notes")

    summary: str = Field(
        description="A concise 3-4 sentence English summary covering ownership, location, plot size, and any active encumbrances or loans."
    )
    full_raw_text: str = Field(
        description="Complete transcription of every word, number, table cell, stamp, and remark found on the document in its original script."
    )

def encode_image(image_path: str) -> str:
    """Helper to convert image to base64 for Groq."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_record(image_path: str) -> LandRecord:
    prompt = (
        "Analyze this Indian land record completely. "
        "1. Parse primary metadata into the structured fields. "
        "2. Provide an analytical summary highlighting land size, owners, and legal remarks/loans. "
        "3. Transcribe the entire document content into 'full_raw_text' so no text or table data is lost."
    )

    # ---------------------------------------------------------
    # ATTEMPT 1: Google Gemini (Primary)
    # ---------------------------------------------------------
    try:
        print("⏳ Attempting extraction with Gemini...")
        gemini_client = genai.Client()
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=LandRecord,
                temperature=0.0
            ),
        )
        return LandRecord.model_validate_json(response.text)

    except Exception as e:
        print(f"⚠️ Gemini failed: {e}\n⏳ Falling back to Groq LLaVA Vision...")

    # ---------------------------------------------------------
    # ATTEMPT 2: Groq Fallback
    # ---------------------------------------------------------
    groq_client = Groq() 
    base64_image = encode_image(image_path)
    
    # Groq requires the schema to be explicitly passed in the prompt for accurate JSON formatting
    schema_str = json.dumps(LandRecord.model_json_schema(), indent=2)
    groq_prompt = f"{prompt}\n\nYou MUST return ONLY a valid JSON object matching this exact schema:\n{schema_str}"

    response = groq_client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": groq_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    
    # Validate and return Groq's JSON output
    return LandRecord.model_validate_json(response.choices[0].message.content)

# Run extraction
record = extract_record("ocr.jpg")

# Save the output to a JSON file
output_file = "extracted_record.json"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(record.model_dump_json(indent=4))

print(f"✅ Success! Data successfully saved to {output_file}")