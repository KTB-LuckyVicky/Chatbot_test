import React, { useState } from 'react';
import './App.css';

function App() {
  const [messageList, setMessageList] = useState([]);
  const [userQuestion, setUserQuestion] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!userQuestion) return;

    const newMessageList = [...messageList, { role: 'user', content: userQuestion }];
    setMessageList(newMessageList);

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/run-python', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userQuestion }),  // 사용자 질문을 Flask 서버에 전송
      });

      const data = await response.json();
      if (data.response) {
        setMessageList([...newMessageList, { role: 'ai', content: data.response }]);
      } else {
        setMessageList([...newMessageList, { role: 'ai', content: 'Error: ' + data.error }]);
      }
    } catch (error) {
      console.error('Error:', error);
      setMessageList([...newMessageList, { role: 'ai', content: 'Error: ' + error.message }]);
    }
    setLoading(false);
    setUserQuestion('');
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>뉴스레터 챗봇</h1>
        <p>뉴스레터를 생성해 드립니다!</p>
      </header>
      <main>
        <div className="chat-box">
          {messageList.map((message, index) => (
            <div key={index} className={`chat-message ${message.role}`}>
              <p>{message.content}</p>
            </div>
          ))}
        </div>
        <form onSubmit={handleSubmit} className="chat-input-form">
          <input
            type="text"
            placeholder="입력한 내용을 바탕으로 뉴스레터를 생성해드릴게요"
            value={userQuestion}
            onChange={(e) => setUserQuestion(e.target.value)}
          />
          <button type="submit" disabled={loading}>
            {loading ? '뉴스레터를 생성하는 중입니다...' : '보내기'}
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;
