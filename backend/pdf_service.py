import os
from datetime import datetime
from sqlalchemy.orm import Session
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as PDFImage
from reportlab.lib.units import inch, cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from .models import Paciente, Chat, Image, ChatMessage
import re
from PIL import Image as PILImage

METRICAS_MODELO = {
    'BG': {'precision': 0.9615, 'recall': 1.0, 'f1-score': 0.9804},
    'D': {'precision': 0.8293, 'recall': 0.7391, 'f1-score': 0.7816},
    'N': {'precision': 0.9259, 'recall': 1.0, 'f1-score': 0.9615},
    'P': {'precision': 0.6429, 'recall': 0.2647, 'f1-score': 0.375},
    'S': {'precision': 0.6667, 'recall': 0.5714, 'f1-score': 0.6154},
    'V': {'precision': 0.6556, 'recall': 0.9516, 'f1-score': 0.7763}
}

traducoes = {
    'BG': 'Background',
    'D': 'Úlcera Diabética', 
    'N': 'Pele Normal',
    'P': 'Úlcera por Pressão',
    'S': 'Ferida Cirúrgica',
    'V': 'Úlcera Venosa'
}

def get_gemini_client():
    """Retorna o cliente do Gemini"""
    try:
        from google import genai
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY não encontrada")
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        raise ImportError("Biblioteca google-genai não instalada")

def extract_probabilities_from_analysis(db: Session, image_hash: str, chat_id: int) -> dict:
    """
    Extrai as probabilidades reais da mensagem de análise do Gemini
    """
    try:
        message = db.query(ChatMessage).filter(
            ChatMessage.chat_id == chat_id,
            ChatMessage.message_type == "analysis",
            ChatMessage.content.like(f'%{image_hash}%')
        ).order_by(ChatMessage.created_at.desc()).first()
        
        if not message:
            return {}
        
        client = get_gemini_client()
        
        prompt = f"""
        Analise o seguinte texto de análise médica e extraia APENAS as probabilidades de classificação 
        no formato EXATO: {{'CLASSE': 'X.XX%', 'CLASSE': 'X.XX%', ...}}
        
        TEXTO:
        {message.content}
        
        Regras:
        1. Extraia apenas as probabilidades no formato de dicionário Python
        2. Use as classes: BG, D, N, P, S, V
        3. Mantenha os valores exatos com duas casas decimais e símbolo de porcentagem
        4. Se não encontrar probabilidades, retorne {{}}
        5. Não adicione nenhum texto explicativo, apenas o dicionário
        6. Somente as que tiverem no texto, não precisa retornar todas as classes se não estiverem presentes
        
        Exemplo 1 de saída esperada:
        {{'BG': '0.01%', 'D': '84.28%', 'N': '0.00%', 'P': '10.50%', 'S': '1.00%', 'V': '4.21%'}}
        
        Exemplo 2 de saída esperada:
        {{'D': '84.28%', 'P': '10.50%', 'S': '1.00%'}}
        """
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        import ast
        try:
            probabilities = ast.literal_eval(response.text.replace('`', '').replace('json', '').strip())
            return probabilities
        except:
            return {}
            
    except Exception as e:
        print(f"❌ Erro ao extrair probabilidades: {e}")
        return {}
    
def get_formal_analysis(image_path: str, classification_data: dict) -> str:
    """
    Usa Gemini para gerar uma análise formal para o PDF
    """
    try:
        client = get_gemini_client()
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        classe_predita = classification_data.get('classe_predita', 'Desconhecida')
        classe_traduzida = classification_data.get('classe_traduzida', 'Desconhecida')
        confianca = classification_data.get('confianca_predita_percentual', 'N/A')
        probabilidades = classification_data.get('probabilidades_completas', {})
        
        prompt = f"""
        Você é um especialista médico criando um laudo formal para documentação clínica.

        DADOS DA CLASSIFICAÇÃO:
        - Classe Predita: {classe_predita} ({classe_traduzida})
        - Confiança do Modelo: {confianca}
        - Probabilidades: {probabilidades}

        Crie uma descrição formal e técnica para inclusão em um laudo médico PDF. A descrição deve:

        1. Ser concisa (máximo 150 palavras)
        2. Usar linguagem médica formal
        3. Descrever as características visuais relevantes
        4. Mencionar o nível de confiança da classificação
        5. Incluir considerações sobre diagnósticos diferenciais baseados nas probabilidades
        6. Manter tom profissional e objetivo

        Formate a resposta em parágrafos curtos, sem marcadores. Não use markdown, apenas texto simples.
        """
        
        from google.genai import types
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[
                prompt,
                types.Part.from_bytes(
                    data=image_data,
                    mime_type='image/jpeg'
                )
            ]
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"❌ Erro ao gerar análise formal: {e}")
        return "Análise formal não disponível no momento."

def create_metrics_table() -> Table:
    """Cria tabela com as métricas do modelo"""
    data = [
        ['Classe', 'Descrição', 'Precision', 'Recall', 'F1-Score']
    ]
    
    for classe, metrics in METRICAS_MODELO.items():
        data.append([
            classe,
            traducoes.get(classe, classe),
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['f1-score']:.3f}"
        ])
    
    table = Table(data, colWidths=[2*cm, 4*cm, 2*cm, 2*cm, 2*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A88')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    return table

def create_image_metrics_table(probabilities: dict) -> Table:
    """Cria tabela com métricas específicas da imagem usando probabilidades reais"""
    sorted_probs = sorted(probabilities.items(), 
                         key=lambda x: float(x[1].rstrip('%')), 
                         reverse=True)
    
    data = [['Classificação', 'Probabilidade', 'Descrição']]
    
    for classe, prob in sorted_probs:
        try:
            prob_value = float(prob.rstrip('%'))
            prob_formatted = f"{prob_value:.2f}%"
        except:
            prob_formatted = prob
        
        data.append([
            classe,
            prob_formatted,
            traducoes.get(classe, classe)
        ])
    
    table = Table(data, colWidths=[2.5*cm, 2.5*cm, 4*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5A88')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F0F8FF')]),
    ]))
    
    return table

def resize_image_for_pdf(image_path: str, max_width: int = 200) -> str:
    """Redimensiona imagem para o PDF e retorna caminho temporário"""
    try:
        with PILImage.open(image_path) as img:
            width_percent = max_width / float(img.size[0])
            new_height = int(float(img.size[1]) * float(width_percent))
            
            img_resized = img.resize((max_width, new_height), PILImage.Resampling.LANCZOS)
            
            temp_path = f"/tmp/{os.path.basename(image_path)}"
            img_resized.save(temp_path, "JPEG", quality=85)
            
            return temp_path
    except Exception as e:
        print(f"❌ Erro ao redimensionar imagem: {e}")
        return image_path

def create_pdf_report(db: Session, paciente_id: int, output_path: str) -> str:
    """
    Cria um relatório PDF profissional para o paciente
    """
    try:
        paciente = db.query(Paciente).filter(Paciente.id == paciente_id).first()
        if not paciente:
            raise ValueError("Paciente não encontrado")
        
        chat = db.query(Chat).filter(Chat.paciente_id == paciente_id).first()
        if not chat:
            raise ValueError("Chat não encontrado")
        
        images = db.query(Image).filter(Image.chat_id == chat.id).all()
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72,
            title=f"Pré-Laudo - {paciente.nome}",
            author="Sistema de Análise de Lesões por Pressão",
            subject=f"Análise de lesões para paciente {paciente.nome}",
            creator="Sistema de Análise Assistida por IA",
            keywords=f"lesão, pressão, diabetes, {paciente.nome}, pré-laudo"
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.HexColor('#2E5A88'),
            spaceAfter=12,
            alignment=1
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2E5A88'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=10,
            textColor=colors.HexColor('#455A64'),
            spaceAfter=8,
            spaceBefore=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            spaceAfter=8,
            leading=13
        )
        
        small_style = ParagraphStyle(
            'CustomSmall',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            spaceAfter=6,
            leading=11
        )
        
        story = []
        
        header_text = """
        <b>PRÉ-LAUDO MÉDICO - ANÁLISE DE LESÕES</b><br/>
        <i>Sistema de IA de Suporte à Decisão Clínica<br/>(IA-CDSS)</i>
        """
        story.append(Paragraph(header_text, title_style))
        story.append(Spacer(1, 10))
        
        data_emissao = f"""
        <b>Data de emissão:</b> {datetime.now().strftime('%d/%m/%Y às %H:%M')}<br/><br/>
        <b>Antonio de Oliveira</b> - Mestrando<br/>
        <b>Ernane Ferreira</b> - Mestrando<br/>
        <b>Prof. Dr. Ricardo Valentim</b>
        """
        story.append(Paragraph(data_emissao, small_style))
        
        story.append(Paragraph("1. INFORMAÇÕES DO PACIENTE", heading_style))
        
        paciente_info = f"""
        <b>Nome:</b> {paciente.nome}<br/>
        <b>Idade:</b> {paciente.idade} anos<br/>
        <b>Sexo:</b> {paciente.sexo}<br/>
        <b>Tipo de Diabetes:</b> {paciente.diabetes_tipo}<br/>
        <b>Documento:</b> {paciente.documento or 'Não informado'}<br/>
        <b>Histórico Médico:</b> {paciente.historico_medico or 'Não informado'}<br/>
        <b>Medicamentos:</b> {paciente.medicamentos or 'Não informado'}<br/>
        <b>Alergias:</b> {paciente.alergias or 'Nenhuma informada'}
        """
        story.append(Paragraph(paciente_info, normal_style))
        
        if images:
            story.append(Paragraph("2. ANÁLISE DAS LESÕES", heading_style))
            
            for i, img in enumerate(images, 1):
                story.append(Paragraph(f"<b>2.{i} Lesão {i}:</b>", subheading_style))
                
                try:
                    if os.path.exists(img.image_path):
                        temp_image_path = resize_image_for_pdf(img.image_path, 400)
                        pdf_image = PDFImage(temp_image_path, width=8*cm, height=6*cm)
                        story.append(pdf_image)
                        story.append(Spacer(1, 10))
                except Exception as e:
                    print(f"❌ Erro ao adicionar imagem: {e}")
                
                img_basic_info = f"""
                <b>Classificação:</b> {img.classification} - {img.description}<br/>
                <b>Arquivo:</b> {img.filename}<br/>
                <b>Distribuição de Probabilidades:</b>
                """
                story.append(Paragraph(img_basic_info, normal_style))
                
                if img.classification != "Pendente" and img.classification != "ERROR":
                    image_hash = os.path.basename(img.image_path).split('.')[0]
                    
                    probabilities = extract_probabilities_from_analysis(db, image_hash, chat.id)
                    
                    if not probabilities:
                        if '(' in img.description and ')' in img.description:
                            confianca = img.description.split('(')[-1].rstrip(')')
                            probabilities = {
                                img.classification: confianca,
                                **{classe: '0.00%' for classe in ['BG', 'D', 'N', 'P', 'S', 'V'] 
                                   if classe != img.classification}
                            }
                        else:
                            probabilities = {img.classification: img.description.split('(')[1].split(')')[0]}
                    
                    metrics_table = create_image_metrics_table(probabilities)
                    story.append(metrics_table)
                    story.append(Spacer(1, 10))
                
                if img.classification != "Pendente" and img.classification != "ERROR":
                    story.append(Paragraph("<b>Análise Formal:</b>", normal_style))
                    
                    classification_data = {
                        'classe_predita': img.classification,
                        'classe_traduzida': img.description.split('(')[0].strip() if '(' in img.description else img.description,
                        'confianca_predita_percentual': img.description.split('(')[-1].rstrip(')') if '(' in img.description else 'N/A',
                        'probabilidades_completas': {img.classification: '100%'}
                    }
                    
                    formal_analysis = get_formal_analysis(img.image_path, classification_data)
                    story.append(Paragraph(formal_analysis, small_style))
        else:
            story.append(Paragraph("Nenhuma imagem analisada disponível.", normal_style))
        
        story.append(Paragraph("3. INFORMAÇÕES DO SISTEMA", heading_style))
        
        story.append(Paragraph("<b>Desempenho do Modelo de Classificação:</b>", subheading_style))
        
        metrics_explanation = """
        <b>Glossário de Métricas:</b><br/>
        • <b>Precision</b>: Acerto nas previsões positivas (exatidão)<br/>
        • <b>Recall</b>: Capacidade de encontrar todos os casos positivos (sensibilidade)<br/>
        • <b>F1-Score</b>: Média harmônica entre Precision e Recall<br/>
        • <b>Valores</b>: 0.000 (pior) a 1.000 (melhor)
        """
        story.append(Paragraph(metrics_explanation, small_style))
        story.append(Spacer(1, 10))
        
        metrics_table = create_metrics_table()
        story.append(metrics_table)
        story.append(Spacer(1, 15))
        
        model_info = """
        <b>Sistema de Classificação:</b> Rede Neural Convolucional VGG16 Fine-Tuned<br/>
        <b>Finalidade:</b> Auxílio à decisão clínica para triagem inicial de lesões<br/>
        <b>Treinamento:</b> Dataset especializado em lesões de pele<br/>
        <b>Processamento:</b> Análise automática de características visuais
        """
        story.append(Paragraph(model_info, normal_style))
        
        story.append(Paragraph("4. OBSERVAÇÕES E RECOMENDAÇÕES", heading_style))
        
        observacoes = """
        • Este documento constitui um <b>PRÉ-LAUDO</b> gerado por sistema de Inteligência Artificial;<br/>
        • <b>Não substitui</b> a avaliação de profissional de saúde qualificado;<br/>
        • Para úlceras por pressão (Classe P), o sistema possui <b>sensibilidade limitada</b> (recall: 26.47%).<br/>
        """
        story.append(Paragraph(observacoes, normal_style))
        
        story.append(Spacer(1, 30))
        footer_text = f"""
        <i>Documento gerado automaticamente - Sistema de Análise de Lesões por Pressão<br/>
        Paciente: {paciente.nome} | ID: {paciente.id} | Emitido em: {datetime.now().strftime('%d/%m/%Y %H:%M')}<br/><br/>
        2025 - UNIVERSIDADE FEDERAL DO RIO GRANDE DO NORTE (UFRN)<br/>
        MESTRADO EM ENGENHARIA ELÉTRICA E DE COMPUTAÇÃO/PPGEEC/CT - NATAL<br/>
        PPGEEC2328 - TÓPICOS ESPECIAIS EM PROCESSAMENTO EMBARCADO E DISTRIBUÍDO
        </i>
        """
        story.append(Paragraph(footer_text, small_style))
        
        doc.build(story)
        
        for img in images:
            try:
                temp_path = f"/tmp/{os.path.basename(img.image_path)}"
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass
                
        return output_path
        
    except Exception as e:
        print(f"❌ Erro ao gerar PDF: {e}")
        raise

def generate_report_for_patient(db: Session, paciente_id: int) -> str:
    """
    Gera relatório PDF e retorna o caminho do arquivo
    """
    reports_dir = "/app/bucket/reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pre_laudo_{paciente_id}_{timestamp}.pdf"
    output_path = os.path.join(reports_dir, filename)
    
    create_pdf_report(db, paciente_id, output_path)
    
    return output_path