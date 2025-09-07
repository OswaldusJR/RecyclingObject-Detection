# ♻️ Recyclable Waste Detection with YOLOv11(s)

This project was developed as part of a **university assignment** .  
It demonstrates how deep learning can be applied to **detect recyclable waste objects** in images, videos, and live camera streams.

The model used is **YOLOv11(s)**, trained on the [TACO dataset](http://tacodataset.org), which contains litter annotations for real-world waste detection tasks.  

===

## 📖 Project Description

- **Model**: YOLOv11(s) (Ultralytics)  
- **Dataset**: TACO (Trash Annotations in Context)  
- **Workflow**:
  1. Dataset preparation & filtering  
  2. Data augmentation & oversampling to balance classes  
  3. Dataset split into training and validation sets  
  4. Training YOLOv11(s) model  
  5. Deployment on Streamlit for interactive demo (image/video upload + live webcam)  

The deployed app allows users to:
- Upload **images or videos** for waste detection  
- Run **real-time webcam for waste detection**  
- Adjust **confidence threshold** and **NMS (overlap filtering)**  
- Filter detections by **object class** (e.g., plastic only)  

Classes are reassigned and mapped into 5 generalized groups from 60 classes
{   'Aluminium foil': 'Metal',
    'Battery': 'Other',
    'Aluminium blister pack': 'Metal',
    'Carded blister pack': 'Other',
    'Other plastic bottle': 'Plastic',
    'Clear plastic bottle': 'Plastic',
    'Glass bottle': 'Glass',
    'Plastic bottle cap': 'Plastic',
    'Metal bottle cap': 'Metal',
    'Broken glass': 'Glass',
    'Food Can': 'Metal',
    'Aerosol': 'Other',
    'Drink can': 'Metal',
    'Toilet tube': 'Other',
    'Other carton': 'Paper',
    'Egg carton': 'Paper',
    'Drink carton': 'Paper',
    'Corrugated carton': 'Paper',
    'Meal carton': 'Paper',
    'Pizza box': 'Paper',
    'Paper cup': 'Paper',
    'Disposable plastic cup': 'Plastic',
    'Foam cup': 'Plastic',
    'Glass cup': 'Glass',
    'Other plastic cup': 'Plastic',
    'Food waste': 'Other',
    'Glass jar': 'Glass',
    'Plastic lid': 'Plastic',
    'Metal lid': 'Metal',
    'Other plastic': 'Plastic',
    'Magazine paper': 'Paper',
    'Tissues': 'Paper',
    'Wrapping paper': 'Paper',
    'Normal paper': 'Paper',
    'Paper bag': 'Paper',
    'Plastified paper bag': 'Paper',
    'Plastic film': 'Plastic',
    'Six pack rings': 'Plastic',
    'Garbage bag': 'Plastic',
    'Other plastic wrapper': 'Plastic',
    'Single-use carrier bag': 'Plastic',
    'Polypropylene bag': 'Plastic',
    'Crisp packet': 'Plastic',
    'Spread tub': 'Plastic',
    'Tupperware': 'Plastic',
    'Disposable food container': 'Plastic',
    'Foam food container': 'Plastic',
    'Other plastic container': 'Plastic',
    'Plastic glooves': 'Plastic',
    'Plastic utensils': 'Plastic',
    'Pop tab': 'Metal',
    'Rope & strings': 'Other',
    'Scrap metal': 'Metal',
    'Shoe': 'Other',
    'Squeezable tube': 'Plastic',
    'Plastic straw': 'Plastic',
    'Paper straw': 'Paper',
    'Styrofoam piece': 'Plastic',
    'Unlabeled litter': 'Other',
    'Cigarette': 'Other'}

    For metrics, you may refer to **models/Additional_info**
===

## 📊 Dataset
This project uses the **TACO (Trash Annotations in Context)** dataset:  
- Website: [http://tacodataset.org](http://tacodataset.org)  
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  

### Citation
```bibtex
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
