from flask import Flask, request, jsonify
from flask_cors import CORS
from test.src.llm123 import get_ai_message

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 모든 출처에 대해 CORS 허용

@app.route('/run-python', methods=['POST'])
def run_python():
    try:
        user_message = request.json.get('message')
        ai_response = get_ai_message(user_message)
        return jsonify({'response': ai_response})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
