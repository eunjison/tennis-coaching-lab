# app.py
"""테니스 코칭 AI 웹 애플리케이션"""
from flask import Flask, request, render_template_string, jsonify
import os
import json
from coaching_engine import full_analysis_pipeline, generate_coaching_report, analyze_demo_frames, get_demo_frames
from frame_extractor import extract_frames, get_demo_frames

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

os.makedirs('uploads', exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🎾 AI 테니스 코치</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f4f8; color: #333; }
        .header { background: linear-gradient(135deg, #1a5276, #2ecc71); color: white; padding: 2rem; text-align: center; }
        .header h1 { font-size: 2rem; margin-bottom: 0.5rem; }
        .header p { opacity: 0.9; }
        .container { max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
        .card { background: white; border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .card h2 { color: #1a5276; margin-bottom: 1rem; }
        .upload-area { border: 2px dashed #bdc3c7; border-radius: 8px; padding: 3rem; text-align: center; cursor: pointer; transition: all 0.3s; }
        .upload-area:hover { border-color: #2ecc71; background: #f8fffe; }
        .btn { background: #2ecc71; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-size: 1rem; cursor: pointer; transition: all 0.3s; }
        .btn:hover { background: #27ae60; }
        .btn:disabled { background: #bdc3c7; cursor: not-allowed; }
        .btn-demo { background: #3498db; }
        .btn-demo:hover { background: #2980b9; }
        .report { white-space: pre-wrap; line-height: 1.8; font-size: 0.95rem; }
        .loading { display: none; text-align: center; padding: 2rem; }
        .loading.active { display: block; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #2ecc71; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto 1rem; }
        @keyframes spin { 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }
        .stat { background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 1.5rem; font-weight: bold; color: #1a5276; }
        .stat-label { font-size: 0.85rem; color: #666; margin-top: 0.25rem; }
        .toggle { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .toggle label { display: flex; align-items: center; gap: 0.5rem; cursor: pointer; padding: 0.5rem 1rem; border-radius: 6px; background: #ecf0f1; }
        .toggle input:checked + span { font-weight: bold; color: #2ecc71; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎾 AI 테니스 코치</h1>
        <p>무인 테니스 연습장 AI 코칭 시스템 | Powered by Azure AI Foundry</p>
    </div>
    
    <div class="container">
        <div class="card">
            <h2>📹 영상 분석</h2>
            <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                <p style="font-size:3rem">📁</p>
                <p>영상 파일을 여기에 드래그하거나 클릭하여 업로드하세요</p>
                <p style="color:#999; margin-top:0.5rem">MP4, AVI, MOV 지원 (최대 500MB)</p>
                <input type="file" id="videoInput" accept="video/*" hidden onchange="handleUpload(this)">
            </div>
            
            <div style="text-align: center; margin-top: 1rem;">
                <span style="color: #999;">또는</span>
            </div>
            
            <div style="text-align: center; margin-top: 1rem;">
                <button class="btn btn-demo" onclick="runDemo()">🎮 데모 분석 실행</button>
            </div>
            
            <div class="toggle" style="margin-top: 1rem; justify-content: center;">
                <label>
                    <input type="checkbox" id="useFinetuned" checked>
                    <span>Fine-tuned 모델 사용</span>
                </label>
            </div>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>🎾 AI 코치가 영상을 분석 중입니다...</p>
            <p style="color:#999; margin-top:0.5rem">프레임 추출 → AI 분석 → 리포트 생성</p>
        </div>
        
        <div class="card" id="resultCard" style="display:none;">
            <h2>📋 코칭 리포트</h2>
            <div class="stats" id="stats"></div>
            <hr style="margin: 1.5rem 0; border: none; border-top: 1px solid #eee;">
            <div class="report" id="report"></div>
        </div>
    </div>
    
    <script>
    async function handleUpload(input) {
        const file = input.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('video', file);
        formData.append('use_finetuned', document.getElementById('useFinetuned').checked);
        
        await analyze(formData, '/analyze');
    }
    
    async function runDemo() {
        const formData = new FormData();
        formData.append('use_finetuned', document.getElementById('useFinetuned').checked);
        
        await analyze(formData, '/demo');
    }
    
    async function analyze(formData, endpoint) {
        document.getElementById('loading').classList.add('active');
        document.getElementById('resultCard').style.display = 'none';
        
        try {
            const response = await fetch(endpoint, { method: 'POST', body: formData });
            const data = await response.json();
            
            document.getElementById('report').textContent = data.report;
            document.getElementById('stats').innerHTML = `
                <div class="stat"><div class="stat-value">${data.model}</div><div class="stat-label">사용 모델</div></div>
                <div class="stat"><div class="stat-value">${data.tokens.toLocaleString()}</div><div class="stat-label">토큰 사용</div></div>
                <div class="stat"><div class="stat-value">$${data.cost}</div><div class="stat-label">예상 비용</div></div>
            `;
            
            document.getElementById('resultCard').style.display = 'block';
        } catch(e) {
            alert('분석 중 오류가 발생했습니다: ' + e.message);
        }
        
        document.getElementById('loading').classList.remove('active');
    }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/demo', methods=['POST'])
def demo_analysis():
    """데모 분석 (영상 없이)"""
    use_finetuned = request.form.get('use_finetuned', 'true') == 'true'
    
    demo_frames = get_demo_frames()
    observations = analyze_demo_frames(demo_frames)
    result = generate_coaching_report(observations, use_finetuned)
    
    cost = round((result['tokens']['total'] / 1_000_000) * 0.50, 4)
    
    return jsonify({
        "report": result["report"],
        "model": "Fine-tuned" if use_finetuned else "Base",
        "tokens": result["tokens"]["total"],
        "cost": f"{cost:.4f}"
    })

@app.route('/analyze', methods=['POST'])
def video_analysis():
    """영상 분석"""
    if 'video' not in request.files:
        return jsonify({"error": "영상 파일이 없습니다."}), 400
    
    video = request.files['video']
    use_finetuned = request.form.get('use_finetuned', 'true') == 'true'
    
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)
    
    try:
        result = full_analysis_pipeline(video_path, use_finetuned)
        cost = round((result['tokens']['total'] / 1_000_000) * 0.50, 4)
        
        return jsonify({
            "report": result["report"],
            "model": "Fine-tuned" if use_finetuned else "Base",
            "tokens": result["tokens"]["total"],
            "cost": f"{cost:.4f}"
        })
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

if __name__ == '__main__':
    print("🎾 테니스 코칭 AI 서버 시작")
    print("http://localhost:5000 으로 접속하세요")
    app.run(host='0.0.0.0', port=5000, debug=True)