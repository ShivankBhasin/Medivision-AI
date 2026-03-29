from utils.image_processor import ImageProcessor
from utils.ocr_engine import MultiEngineOCR
from utils.llm_postprocessor import LLMPostProcessor
import fitz
from PIL import Image
import io
import os
from typing import List, Dict, Tuple

class PrescriptionOCR:
    def __init__(self, groq_api_key: str):
        self.preprocessor = ImageProcessor()
        self.ocr_engine = MultiEngineOCR()
        self.llm_processor = LLMPostProcessor(groq_api_key)

    def process(self, file_path: str) -> Dict:
        if file_path.lower().endswith('.pdf'):
            images = self._pdf_to_images(file_path)
            results = [self._process_single_image(img) for img in images]
            return self._merge_page_results(results)
        else:
            return self._process_single_image(file_path)
        
    def _pdf_to_images(self, pdf_path: str) -> List:
        doc = fitz.open(pdf_path)
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))

            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)

        doc.close()
        return images
    
    def _process_single_image(self, image_input) -> Dict:
        preprocessed, original = self.preprocessor.preprocess(image_input)

        ocr_output = self.ocr_engine.extract_text(preprocessed)

        final_output = self.llm_processor.clean_and_structure(ocr_output)

        final_output['ocr raw'] = ocr_output
        final_output['processing_metadata'] = {
            'ocr_confidence': ocr_output['confidence'],
            'engines_used': list(ocr_output['individual_engines'].keys())
        }

        return final_output
    
    def _merge_page_results(self, results: List[Dict]) -> Dict:
        if len(results) == 1:
            return results[0]
        
        merged = results[0].copy()

        for result in results[1:]:
            if result['medications']:
                merged['medications'].extend(result['medications'])

        return merged