import os
import sys
import traceback
from celery import current_task
from .celery_app import celery_app

sys.path.append('/app/backend')

def classificar_imagem_batch():
    """Task Celery para classificar todas as imagens pendentes no banco de dados"""
    try:
        from database import SessionLocal
        from models import Image
        from classification_model import classificar_imagem
        
        print("🎯 STARTING BATCH PROCESSING")
        
        db = SessionLocal()
        
        try:
            imagens = db.query(Image).filter(Image.classification == "Pendente").all()
            print(f"📊 {len(imagens)} images pending")
            
            resultados = []
            for i, img in enumerate(imagens, 1):
                print(f"\n--- [{i}/{len(imagens)}] {img.filename} ---")
                
                try:
                    image_path = img.image_path
                    print(f"📂 Relative path: {image_path}")
                    
                    if not image_path.startswith('/'):
                        image_path = f"/app/{image_path}"
                    
                    print(f"📂 Absolute path: {image_path}")
                    
                    if not os.path.exists(image_path):
                        print(f"❌ File not found at: {image_path}")
                        img.classification = "ERROR"
                        img.description = "File not found"
                        db.commit()
                        resultados.append({
                            'image_id': img.id,
                            'filename': img.filename,
                            'status': 'error',
                            'error': 'File not found'
                        })
                        continue
                    
                    resultado = classificar_imagem(image_path)
                    
                    if resultado:
                        img.classification = resultado['classe']
                        img.description = f"{resultado['translated_class']} ({resultado['confianca']})"
                        db.commit()
                        print(f"✅ Atualizado: {resultado['classe']}")
                        resultados.append({
                            'image_id': img.id,
                            'filename': img.filename,
                            'status': 'success',
                            'classification': resultado['classe'],
                            'description': resultado['translated_class'],
                            'confidence': resultado['confianca']
                        })
                    else:
                        print(f"❌ Failed to classify image {img.filename}")
                        img.classification = "ERROR"
                        img.description = "Failed to classify image"
                        db.commit()
                        resultados.append({
                            'image_id': img.id,
                            'filename': img.filename,
                            'status': 'error',
                            'error': 'Failed to classify image'
                        })
                        
                except Exception as img_error:
                    error_msg = f"Error in processing: {str(img_error)}"
                    print(f"💥 ERROR in image {img.filename}: {error_msg}")
                    print(traceback.format_exc())
                    
                    img.classification = "ERROR"
                    img.description = error_msg[:255]
                    db.commit()
                    resultados.append({
                        'image_id': img.id,
                        'filename': img.filename,
                        'status': 'error',
                        'error': error_msg
                    })
                    continue
            
            print(f"\n🎉 COMPLETED! {len([r for r in resultados if r.get('status') == 'success'])} images processed successfully")
            print(f"❌ {len([r for r in resultados if r.get('status') == 'error'])} images with error")
            
            return {
                'status': 'completed',
                'total_processed': len(resultados),
                'success_count': len([r for r in resultados if r.get('status') == 'success']),
                'error_count': len([r for r in resultados if r.get('status') == 'error']),
                'results': resultados
            }
            
        except Exception as e:
            error_msg = f"Error in general processing: {str(e)}"
            print(f"💥 GENERAL ERROR: {error_msg}")
            print(traceback.format_exc())
            return {
                'status': 'error',
                'error': error_msg,
                'results': []
            }
        finally:
            db.close()
            
    except ImportError as e:
        error_msg = f"Error in import: {str(e)}"
        print(f"💥 IMPORT ERROR: {error_msg}")
        print(traceback.format_exc())
        return {
            'status': 'error',
            'error': error_msg,
            'results': []
        }

@celery_app.task(bind=True, name='app.tasks.processar_imagens_pendentes')
def processar_imagens_pendentes(self):
    """Celery task to process all pending images in the database"""
    return classificar_imagem_batch()

@celery_app.task(bind=True, name='app.tasks.classificar_imagem_individual')
def classificar_imagem_individual(self, image_id):
    """Celery task to classify an individual image and generate a description"""
    try:
        from database import SessionLocal
        from models import Image, ChatMessage
        from classification_model import classificar_imagem
        from image_description_service import describe_image_with_analysis
        
        db = SessionLocal()
        
        try:
            img = db.query(Image).filter(Image.id == image_id).first()
            if not img:
                return {
                    'status': 'error',
                    'error': f'Image with ID {image_id} not found'
                }
            
            print(f"🎯 Processing individual image: {img.filename} (ID: {image_id})")
            
            image_path = img.image_path
            if not image_path.startswith('/'):
                image_path = f"/app/{image_path}"
            
            if not os.path.exists(image_path):
                error_msg = f"File not found: {image_path}"
                img.classification = "ERROR"
                img.description = error_msg
                db.commit()
                return {
                    'status': 'error',
                    'error': error_msg
                }
            
            resultado = classificar_imagem(image_path)
            
            if resultado.get('status') == 'sucesso':
                img.classification = resultado['predicted_class']
                img.description = f"{resultado['translated_class']} ({resultado['predicted_percentage_confidence']})"
                
                try:
                    print(f"📝 Generating technical description for image {img.id}...")
                    descricao_tecnica = describe_image_with_analysis(image_path, resultado)
                    
                    mensagem_chat = ChatMessage(
                        chat_id=img.chat_id,
                        content=f"""
Image classified as {resultado['translated_class']} with a probability of {resultado['predicted_percentage_confidence']}.

@@IMAGE:{os.path.basename(image_path).split('.')[0]}@@

{descricao_tecnica}
""",
                        is_user=False,
                        message_type="analysis"
                    )
                    db.add(mensagem_chat)
                    print(f"✅ Descrição técnica e mensagem criadas para imagem {img.id}")
                    
                except Exception as desc_error:
                    print(f"⚠️  Error generating technical description: {desc_error}")
                    mensagem_chat = ChatMessage(
                        chat_id=img.chat_id,
                        content=f"""
Image classified as {resultado['translated_class']} with a probability of {resultado['predicted_percentage_confidence']}.

$$IMAGE:{os.path.basename(image_path).split('.')[0]}$$

*Technical analysis not available at the moment.*
                        """,
                        is_user=False,
                        message_type="analysis"
                    )
                    db.add(mensagem_chat)
                
                db.commit()
                
                return {
                    'status': 'success',
                    'image_id': image_id,
                    'filename': img.filename,
                    'classification': resultado['predicted_class'],
                    'description': resultado['translated_class'],
                    'confidence': resultado['predicted_percentage_confidence'],
                    'technical_analysis_created': True
                }
            else:
                error_msg = resultado.get('message', 'Falha na classificação da imagem')
                img.classification = "ERROR"
                img.description = error_msg
                db.commit()
                return {
                    'status': 'error',
                    'error': error_msg
                }
                
        except Exception as e:
            error_msg = f"Error in processing: {str(e)}"
            print(f"💥 ERROR: {error_msg}")
            print(traceback.format_exc())
            
            try:
                img.classification = "ERROR"
                img.description = error_msg[:255]
                db.commit()
            except:
                pass
            
            return {
                'status': 'error',
                'error': error_msg
            }
        finally:
            db.close()
            
    except ImportError as e:
        error_msg = f"Error in import: {str(e)}"
        print(f"💥 IMPORT ERROR: {error_msg}")
        return {
            'status': 'error',
            'error': error_msg
        }