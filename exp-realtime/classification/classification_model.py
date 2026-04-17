import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

MODEL_H5_PATH = '/Users/ernane/me/diabetes-lesion-analysis/backend/models/best_wound_classifier_FINETUNED.h5'
METRICS_CSV_PATH = '/Users/ernane/me/diabetes-lesion-analysis/backend/models/wound_metrics_report_FINETUNED.csv'

CLASSES = ['BG', 'D', 'N', 'P', 'S', 'V']
IMG_SIZE = (224, 224)

MODEL = None
METRICS_DF = None

def carregar_recursos():
    """Carrega modelo e métricas"""
    global MODEL, METRICS_DF
    
    if MODEL is not None:
        return True

    try:
        print("📦 Loading Model...")
        
        base_model = VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(len(CLASSES), activation='softmax')(x)
        MODEL = Model(inputs=base_model.input, outputs=predictions)
        
        MODEL.load_weights(MODEL_H5_PATH)
        print("✅ Model loaded")
        
        METRICS_DF = pd.read_csv(METRICS_CSV_PATH, index_col=0)
        print("✅ Metrics loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        return False

def traduzir_classe(classe):
    translators = {
        'BG': 'Background',
        'D': 'Diabetic Ulcer', 
        'N': 'Normal Skin',
        'P': 'Pressure Ulcer',
        'S': 'Surgical Wound',
        'V': 'Venous Ulcer'
    }
    return translators.get(classe, classe)

def classificar_imagem(image_path: str) -> dict:
    """
    Classification of the image using the fine-tuned VGG16 model. Returns a dictionary with the predicted class, confidence
    """
    if not carregar_recursos():
        return {"status": "erro", "message": "Failed to load model or metrics."}
    
    try:
        print(f"🔍 Processing: {os.path.basename(image_path)}")
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = MODEL.predict(img_array, verbose=0)[0] # Get the first (and only) prediction
        
        class_idx = np.argmax(predictions)
        predicted_class = CLASSES[class_idx]
        predicted_confidence = float(predictions[class_idx])
        
        odds = {c: f"{p*100:.2f}%" for c, p in zip(CLASSES, predictions)}
        
        recall_p = float(METRICS_DF.loc['P', 'recall'])
        
        top_classes = np.argsort(predictions)[::-1] # Indexes of classes sorted by confidence
        top_3_classes = [CLASSES[i] for i in top_classes[:3]]
        
        dados_analise = {
            "status": "success",
            "predicted_class": predicted_class,
            "predicted_percentage_confidence": f"{predicted_confidence*100:.2f}%",
            "translated_class": traduzir_classe(predicted_class),
            "complete_probabilities": odds,
            "top_3_classes": top_3_classes,
            "metric_f1_predicted_class": float(METRICS_DF.loc[predicted_class, 'f1-score']),
            "risk_p": {
                "Recall_P": recall_p,
                "Warning_P": f"Recall histórico ({recall_p:.2f}) para Úlcera por Pressão é baixo. Cautela é necessária."
            }
        }
        
        print(f"✅ Result: {dados_analise['predicted_class']} ({dados_analise['predicted_percentage_confidence']})")
        return dados_analise
        
    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return {"status": "error", "message": str(e)}