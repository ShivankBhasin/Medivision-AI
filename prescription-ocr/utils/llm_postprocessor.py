from groq import Groq
import json
import os
from typing import List, Dict, Tuple

class LLMPostProcessor:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key = groq_api_key) if groq_api_key else None

    def clean_and_structure(self, ocr_output: Dict) -> Dict:
        raw_text = ocr_output['raw_text']
        partial_structured = ocr_output['structured']

        prompt = self._build_prompt(raw_text, partial_structured)
        cleaned_data = self._call_llm(prompt) if self.client else self._fallback_structure()

        return cleaned_data
    
    def _build_prompt(self, raw_text: str, partial: Dict) -> str:
        return f"""You are a medical data extraction expert. You recieve OCR output from handwritten medical prescriptions. Your job is to extract and structure the information accurately.

OCR Raw Text:
{raw_text}

Partially Extracted Data:
{json.dumps(partial, indent=2)}

Extract and return a JSON object with this exact structure:
{{
  "patient_info": {{
    "name": "Patient full name",
    "age": "Age as integer",
    "gender": "M/F/Other or null",
    "date": "Date of prescription in YYYY-MM--DD format or null"
  }},
  "doctor_info":{{
    "name": "Doctor's name",
    "registration_number": "Medical registration number if visible",
    "specialization": "Specialization if mentioned"
  }},
  "medications": [
    {{
      "name": "Medication name",
      "dosage": "Dosage (e.g., 500mg)",
      "frequency": "Frequency (e.g., 1-0-1, BD, TID)",
      "duration": "Duration (e.g., 5 days, 1 week)",
      "instructions": "Special instructions (e.g., after food)"
    }}
  ],
  "diagnosis": "Diagnosis or chief complaint if mentioned",
  "additional_instructions": "Any additional instructions or advice",
  "confidence_score": "Your confidence in this extraction (0-100)"
}}

Rules: 
1. If information is not clearly visible, use null
2. Correct obvious OCR errors (e.g., "Parace1amol" -> "Paracetamol)
3. Standardize medication frequencies (e.g., "twice daily" -> "BD")
4. Extract only information that is actually present
5. Return ONLY valid JSON, no additional text

JSON Output: """
    
    def _call_llm(self, prompt: str) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "Your are a medical data extraction expert. You always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            response_text = response.choices[0].message.content.strip()

            if response_text.startswith('```json'):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith('```'):
                response_text = response_text[3:-3].strip()

            cleaned_data = json.loads(response_text)

            return cleaned_data

        except json.JSONDecodeError as e:
            print(f"LLM returned invalid JSON: {e}")
            return self._fallback_structure()
        
        except Exception as e:
            print(f"LLM processing failed: {e}")
            return self._fallback_structure()
        
    def _fallback_structure(self) -> Dict:
        return{
            "patient_info": {"name": None, "age": None, "gender": None, "date": None},
            "doctor_info": {"name": None, "registration_number": None, "specialization": None},
            "medications": [],
            "diagnosis": None,
            "additional_instructions": None,
            "confidence_score": 0
        }