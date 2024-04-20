// WebcamCapture.tsx
import React, { useRef } from 'react';
import Webcam from 'react-webcam';

const WebcamCapture = () => {
  const webcamRef = useRef(null);

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    sendImageToServer(imageSrc);
  };

  const sendImageToServer = (imageSrc) => {
    fetch('http://127.0.0.1:5000', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ image: imageSrc })
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));
  };

  return (
    <div>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width="100%"
      />
      <button onClick={capture}>Capture photo</button>
    </div>
  );
};

export default WebcamCapture;
