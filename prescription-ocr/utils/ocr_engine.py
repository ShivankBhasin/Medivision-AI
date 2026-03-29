from PIL import Image
import easyocr
import pytesseract  
from paddleocr import PaddleOCR 
import numpy as np
from typing import List, Dict, Tuple, Optional
import platform

if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  

class MultiEngineOCR:
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False, enable_mkldnn=False)
        self.engines_loaded = True

    def extract_text(self, preprocessed_image) -> Dict:
        results = {
            'easyocr': self._run_easyocr(preprocessed_image),
            'tesseract': self._run_tesseract(preprocessed_image),
            'paddleocr': self._run_paddleocr(preprocessed_image)
        }
        
        merged_text = self._merge_results(results)
        structured_data = self._extract_structured_info(merged_text)

        return {
            'raw_text': merged_text,
            'structured': structured_data,
            'individual_engines': results,
            'confidence': self._calculate_confidence(results)
        }
    
    def _run_easyocr(self, img) -> str:
        try:
            results = self.easyocr_reader.readtext(img)
            text_lines = [text for (bbox, text, conf) in results if conf > 0.3]

            return '\n'.join(text_lines)
        except Exception as e:
            print(f"EasyOCR failed: {e}")
            return ""
        
    def _run_tesseract(self, img) -> str:
        try:
            config = '--psm 6 --oem 3'
            text = pytesseract.image_to_string(img, config=config)  

            return text.strip()
        except Exception as e:
            print(f"Tesseract failed: {e}")
            return ""
        
    def _run_paddleocr(self, img) -> str:
        try:
            result = self.paddle_ocr.ocr(img, cls=True)

            if result is None or len(result) == 0:
                return ""
            
            text_lines = []
            for line in result[0]:
                if line[1][1] > 0.5:
                    text_lines.append(line[1][0])

            return '\n'.join(text_lines)
        except Exception as e:
            print(f"PaddleOCR failed: {e}")
            return ""
        
    def _merge_results(self, results: Dict[str, str]) -> str:
        all_lines = []

        for engine, text in results.items():
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                all_lines.extend(lines)

        unique_lines = list(dict.fromkeys(all_lines))

        return '\n'.join(unique_lines)
    
    def _extract_structured_info(self, text: str) -> Dict:
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        structured = {
            'patient_name': None,
            'age': None,
            'date': None,
            'medications': [],
            'diagnosis': None,
            'doctor_name': None,
            'raw_lines': lines
        }

        keywords = {
            'patient': ['patient', 'name', 'pt.', 'pt'],
            'age': ['age', 'yrs', 'years'],
            'date': ['date', 'dt.'],
            'diagnosis': ['diagnosis', 'dx', 'complaint'],
            'medication': ['rx', 'medication', 'medicine', 'tab', 'cap', 'syp'],
            'doctor': ['dr.', 'doctor', 'signature']
        }

        for i, line in enumerate(lines):
            line_lower = line.lower()

            if any(kw in line_lower for kw in keywords['patient']) and not structured['patient_name']:
                structured['patient_name'] = self._extract_value_after_keyword(line, keywords['patient'])

            if any(kw in line_lower for kw in keywords['age']) and not structured['age']:
                structured['age'] = self._extract_number(line)

            if any(kw in line_lower for kw in keywords['medication']):
                medication = self._parse_medication_line(line, lines[i:i+3] if i+3 < len(lines) else lines[i:])
                if medication:
                    structured['medications'].append(medication)

        return structured
    
    def _extract_value_after_keyword(self, line: str, keywords: List[str]) -> str:
        line_lower = line.lower()
        for kw in keywords:
            if kw in line_lower:
                parts = line.split(':')
                if len(parts) > 1:
                    return parts[1].strip()
                
                idx = line_lower.find(kw)
                value = line[idx + len(kw):].strip()

                value = ''.join(filter(lambda x: x not in ':-', value)).strip()
                return value if value else None
        return None  
        
    def _extract_number(self, line: str) -> int:
        import re
        numbers = re.findall(r'\d+', line)
        return int(numbers[0]) if numbers else None
    
    def _parse_medication_line(self, line: str, context_lines: List[str]) -> Dict:
        medication = {
            'name': None,
            'dosage': None,
            'frequency': None,
            'duration': None
        }

        import re

        med_pattern = r'(tab|cap|syp|inj)\.?\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)?)'  
        match = re.search(med_pattern, line, re.IGNORECASE)
        if match:
            medication['name'] = match.group(2).strip()

        dosage_pattern = r'(\d+\s*(?:mg|ml|mcg))'
        dosage_match = re.search(dosage_pattern, line, re.IGNORECASE)
        if dosage_match:
            medication['dosage'] = dosage_match.group(1)

        frequency_patterns = ['1-0-1', '1-1-1', '0-0-1', 'bd', 'tid', 'qid', 'sos']
        for pattern in frequency_patterns:
            if pattern in line.lower():
                medication['frequency'] = pattern
                break

        duration_pattern = r'(\d+\s*(?:days?|weeks?|months?))'
        duration_match = re.search(duration_pattern, line, re.IGNORECASE)
        if duration_match:
            medication['duration'] = duration_match.group(1)

        return medication if medication['name'] else None
    
    def _calculate_confidence(self, results: Dict[str, str]) -> float:
        engines_with_results = sum(1 for text in results.values() if text.strip())
        confidence = (engines_with_results / len(results)) * 100

        return round(confidence, 2)