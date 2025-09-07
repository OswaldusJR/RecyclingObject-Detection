# ♻️ Recyclable Waste Detection with YOLOv11(s)

This project was developed as part of a **university assignment** .  
It demonstrates how deep learning can be applied to **detect recyclable waste objects** in images, videos, and live camera streams.

The model used is **YOLOv11(s)**, trained on the [TACO dataset](http://tacodataset.org), which contains litter annotations for real-world waste detection tasks.  

You may tryout the app via "recyclingobjectdetection.streamlit.app"

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

Classes are reassigned and mapped into 5 generalized groups from 60 classes of specific items

class_mapping = {
    'Plastic': 0,
    'Metal': 1,
    'Glass': 2,
    'Paper': 3,
    'Other': 4
}

    For metrics, you may refer to "models/Additional_info"

## 📊 Dataset
This project uses the **TACO (Trash Annotations in Context)** dataset:  
- Website: [http://tacodataset.org](http://tacodataset.org)  
- License: [MIT]

### Citation
```bibtex
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
