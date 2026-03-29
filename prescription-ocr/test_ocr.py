from prescription_ocr import PrescriptionOCR
import json
import os
from dotenv import load_dotenv

load_dotenv()

def test_prescription_ocr():
    groq_key = os.getenv('GROQ_API_KEY')

    ocr_system = PrescriptionOCR(groq_api_key=groq_key)

    test_file = 'test_images/test_prescription5.png'

    print(f"Processing: {test_file}")
    print("-" * 50)

    result = ocr_system.process(test_file)

    print("\n Extraction Complete!\n")
    print(json.dumps(result, indent=2))

    with open('output.json', 'w') as f:
        json.dump(result, f, indent=2)

    print("\n Results saved to output.json")

if __name__ == "__main__":
    test_prescription_ocr()