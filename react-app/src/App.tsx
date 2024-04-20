import React from 'react';
import WebcamCapture from './WebcamCapture';

import './WebcamStyles.css'
const App: React.FC = () => {
  return (
      <div className="App">
          <h1>Webcam Capture</h1>
          <WebcamCapture />
      </div>
  );
};

export default App;