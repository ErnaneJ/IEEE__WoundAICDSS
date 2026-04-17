import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

MODEL_H5_PATH = '/app/backend/models/best_wound_classifier_FINETUNED.h5'
METRICS_CSV_PATH = '/app/backend/models/wound_metrics_report_FINETUNED.csv'

CLASSES = ['BG', 'D', 'N', 'P', 'S', 'V']
IMG_SIZE = (224, 224)

MODEL = None
METRICS_DF = None

def carregar_recursos():
    """Load the model and metrics if they are not already loaded."""
    global MODEL, METRICS_DF
    
    if MODEL is not None:
        return True

    try:
        print("📦  Loading Model....")
        
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
        print(f"❌ Error to load model or metrics: {e}")
        return False

def traduzir_classe(classe):
    traducoes = {
        'BG': 'Background',
        'D': 'Diabetic Ulcer', 
        'N': 'Normal Skin',
        'P': 'Pressure Ulcer',
        'S': 'Surgical Wound',
        'V': 'Venous Ulcer'
    }
    return traducoes.get(classe, classe)

def classificar_imagem(image_path: str) -> dict:
    """
    Classifies a wound image using the pre-trained model and returns the predicted class, confidence, and relevant metrics.
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
        confianca_predita = float(predictions[class_idx])
        
        probabilidades = {c: f"{p*100:.2f}%" for c, p in zip(CLASSES, predictions)}
        
        recall_p = float(METRICS_DF.loc['P', 'recall'])
        
        top_classes = np.argsort(predictions)[::-1]
        top_3_classes = [CLASSES[i] for i in top_classes[:3]]
        
        dados_analise = {
            "status": "success",
            "predicted_class": predicted_class,
            "predicted_percentage_confidence": f"{confianca_predita*100:.2f}%",
            "translated_class": traduzir_classe(predicted_class),
            "complete_probabilities": probabilidades,
            "top_3_classes": top_3_classes,
            "metric_f1_predicted_class": float(METRICS_DF.loc[predicted_class, 'f1-score']),
            "risk_p": {
                "Recall_P": recall_p,
                "Warning_P": f"Historical recall ({recall_p:.2f}) for Pressure Ulcer is low. Caution is advised."
            }
        }
        
        print(f"✅ Result: {dados_analise['predicted_class']} ({dados_analise['predicted_percentage_confidence']})")
        return dados_analise
        
    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return {"status": "erro", "message": str(e)}