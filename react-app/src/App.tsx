// App.tsx
import React, { useState } from 'react';
import io from 'socket.io-client';
import './App.css';
import mogbotImage from './mogbot.png';

const socket = io('http://127.0.0.1:5000');



function App() {
  const [frame, setFrame] = useState<string>('');
  const [streaming, setStreaming] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(true);  // State to manage modal visibility



  const startStreaming = () => {
      socket.on('frame', (data: { data: string }) => {
          setFrame(data.data);
      });
      socket.emit('start_stream');
      setStreaming(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };

  function handleButtonClick() {
    startStreaming();
    closeModal();
}
  
  return (
    <div className="app-container">
        <div style={{ width: '50vw', height: '50vh', overflow: 'hidden', display: streaming ? 'block' : 'none' }}>
            <img src={`data:image/jpeg;base64,${frame}`} alt="Webcam Stream" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        </div>
        {isModalOpen && (
          <div className='ok'>
            <h2 className='mogulator'>Mogulator</h2>
            <img src={mogbotImage} alt="Mogbot" style={{ width: '500px' , margin: '-130px'}} />
            <p className='mogdes'>Mogcopter performs a mog-by to use our mogulator to mogulate your mog score</p>
            <button className="modal-close-button" onClick={handleButtonClick}>START MOG ANALYSIS</button>         
          </div>
        )}
    </div>
);
}

export default App;
