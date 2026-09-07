import os
import json
import base64
import re
from typing import Optional
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from groq import Groq
import warnings
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify keys are loaded
if not os.environ.get("GEMINI_API_KEY"):
    print("⚠️ Warning: GEMINI_API_KEY not found in environment or .env file.")
if not os.environ.get("GROQ_API_KEY"):
    print("⚠️ Warning: GROQ_API_KEY not found in environment or .env file.")

# Silence Google GenAI SDK warnings
warnings.filterwarnings("ignore", message="Direct use of automatic function calling")
logging.getLogger("google_genai").setLevel(logging.ERROR)

# 1. Define the extraction schema (Made full_raw_text Optional)
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
    full_raw_text: Optional[str] = Field(
        default=None, 
        description="Complete transcription of every word. Will be null if using fallback model."
    )

def encode_image(image_path: str) -> str:
    """Helper to convert image to base64 for Groq."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_record(image_path: str) -> LandRecord:
    # ---------------------------------------------------------
    # ATTEMPT 1: Google Gemini (Primary - FULL TEXT)
    # ---------------------------------------------------------
    gemini_prompt = (
        "Analyze this Indian land record completely. "
        "1. Parse primary metadata into the structured fields. "
        "2. Provide an analytical summary highlighting land size, owners, and legal remarks/loans. "
        "3. Transcribe the entire document content into 'full_raw_text' so no text or table data is lost."
    )
    
    try:
        print("⏳ Attempting extraction with Gemini...")
        # The client automatically picks up os.environ["GEMINI_API_KEY"]
        gemini_client = genai.Client()
        
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                gemini_prompt
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
    # ATTEMPT 2: Groq Fallback (NO FULL TEXT to save tokens)
    # ---------------------------------------------------------
    # The client automatically picks up os.environ["GROQ_API_KEY"]
    groq_client = Groq() 
    base64_image = encode_image(image_path)
    schema_str = json.dumps(LandRecord.model_json_schema(), indent=2)
    
    # We tell Groq specifically NOT to transcribe the text
    groq_prompt = (
        "Analyze this Indian land record completely.\n"
        "CRITICAL ABORT CONDITION: If the image is NOT a land record (e.g., it is an essay, random photo, or receipt), "
        "you must IMMEDIATELY skip all analysis. DO NOT output a <think> block. IMMEDIATELY output the JSON with 'N/A' for all string fields and empty arrays.\n\n"
        "1. Parse primary metadata into the structured fields.\n"
        "2. Provide an analytical summary.\n\n"
        f"You MUST return ONLY a valid JSON object matching this exact schema:\n{schema_str}\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Output ONLY valid, raw JSON.\n"
        "2. TOKEN LIMIT WARNING: You MUST set the 'full_raw_text' field to null. Do NOT transcribe the document.\n"
        "3. DO NOT output a <think> block. Skip all reasoning.\n"
        "4. Do not use markdown formatting (no ```json backticks).\n"
        "5. Your response must start exactly with the '{' character and end with '}'."
    )

    response = groq_client.chat.completions.create(
        model="qwen/qwen3.6-27b",
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
        max_completion_tokens=900,
        top_p=0.95,
        stream=False,
        stop=None
    )
    
    # Clean up the output robustly
    raw_content = response.choices[0].message.content or ""
    
    # Safely remove <think> blocks
    raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
    if '<think>' in raw_content:
        raw_content = raw_content.split('<think>')[-1]
        
    raw_content = raw_content.replace("```json", "").replace("```", "").strip()
    
    start_idx = raw_content.find('{')
    end_idx = raw_content.rfind('}')
    
    if start_idx != -1 and end_idx > start_idx:
        cleaned_content = raw_content[start_idx:end_idx+1]
        return LandRecord.model_validate_json(cleaned_content)
    else:
        # SAFETY NET: If the model ran out of tokens panicking over an invalid image,
        # we catch it here and return a hardcoded "N/A" record instead of crashing.
        print("⚠️ Groq output was cut off or invalid. Generating default N/A record.")
        return LandRecord(
            state="N/A",
            district="N/A",
            taluka_or_tehsil="N/A",
            village="N/A",
            survey_or_gat_no="N/A",
            khata_no="N/A",
            total_area="N/A",
            owners=[],
            liabilities_or_loans=[],
            summary="Invalid image. The uploaded document could not be processed or is not a recognizable land record.",
            full_raw_text=None
        )

# Run extraction when executed directly
if __name__ == "__main__":
    record = extract_record("ocr.jpg")

    # Save the output to a JSON file
    output_file = "extracted_record.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(record.model_dump_json(indent=4))

    print(f"✅ Success! Data successfully saved to {output_file}")