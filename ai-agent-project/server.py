"""
FastAPI 서버 - FIXED VERSION
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from dotenv import load_dotenv

from modules.pipeline import SmartRAGPipeline

load_dotenv()

app = FastAPI(title="Smart RAG with PLoS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = SmartRAGPipeline(
    rag_db_path=os.getenv("RAG_DB_PATH", "./data/pharmacology-DB"),
    use_claude=True,
    claude_api_key=os.getenv("ANTHROPIC_API_KEY")
)

class QueryRequest(BaseModel):
    question: str
    persona: Optional[str] = "basic"

@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    try:
        result = pipeline.process(
            question=request.question,
            persona_key=request.persona
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/personas")
async def get_personas():
    return JSONResponse(content=pipeline.persona_engine.templates)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """
    동적 페르소나 로딩 UI
    """
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart RAG Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
        }
        h1 { color: #333; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select, textarea { 
            width: 100%; 
            padding: 10px; 
            font-size: 16px; 
        }
        button {
            background: #667eea;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover { background: #5568d3; }
        .result { 
            margin-top: 30px; 
            padding: 20px; 
            background: #f5f5f5; 
            border-radius: 5px; 
        }
        .loading { display: none; color: #667eea; }
        .loading.active { display: block; }
    </style>
</head>
<body>
    <h1>🎭 AI Agent RAG with PLoS API</h1>
    
    <div class="form-group">
        <label>Persona:</label>
        <select id="persona">
            <option value="">Loading personas...</option>
        </select>
    </div>
    
    <div class="form-group">
        <label>Question:</label>
        <textarea id="question" rows="3" placeholder="예: 암 치료에 아스피린이 도움되나요?"></textarea>
    </div>
    
    <button onclick="ask()">Ask Question</button>
    
    <div class="loading" id="loading">🔄 Processing...</div>
    
    <div class="result" id="result" style="display:none;">
        <h3>Response:</h3>
        <p id="answer"></p>
        <hr>
        <small>
            <strong>Source:</strong> <span id="source"></span><br>
            <strong>Intent:</strong> <span id="intent"></span>
        </small>
    </div>

    <script>
        // ✅ 페르소나 동적 로딩!
        async function loadPersonas() {
            try {
                console.log('📥 Loading personas from API...');
                const response = await fetch('/api/personas');
                const personas = await response.json();
                
                console.log('✅ Personas loaded:', Object.keys(personas));
                
                const select = document.getElementById('persona');
                select.innerHTML = '<option value="">-- Select Persona --</option>';
                
                // 동적으로 옵션 추가!
                for (const [key, data] of Object.entries(personas)) {
                    const option = document.createElement('option');
                    option.value = key;
                    option.textContent = `${data.name} - ${data.description}`;
                    select.appendChild(option);
                }
                
                console.log(`✅ ${Object.keys(personas).length} personas loaded in dropdown`);
            } catch (error) {
                console.error('❌ Failed to load personas:', error);
                alert('Failed to load personas: ' + error);
            }
        }
        
        async function ask() {
            const question = document.getElementById('question').value;
            const persona = document.getElementById('persona').value;
            
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            if (!persona) {
                alert('Please select a persona');
                return;
            }
            
            document.getElementById('loading').classList.add('active');
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question, persona})
                });
                
                const data = await response.json();
                
                document.getElementById('answer').textContent = data.answer;
                document.getElementById('source').textContent = data.source_type;
                document.getElementById('intent').textContent = data.intent.category;
                document.getElementById('result').style.display = 'block';
                
            } catch (error) {
                alert('Error: ' + error);
            } finally {
                document.getElementById('loading').classList.remove('active');
            }
        }
        
        // ✅ 페이지 로드 시 자동 실행!
        window.addEventListener('DOMContentLoaded', () => {
            console.log('🚀 Page loaded, loading personas...');
            loadPersonas();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    print("="*70)
    print("🚀 Smart RAG Server Starting...")
    print("="*70)
    print(f"\n📍 Open: http://localhost:88888")
    print(f"📚 API Docs: http://localhost:8888/docs\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8888)