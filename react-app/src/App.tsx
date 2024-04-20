import React, { useRef } from 'react';
import Webcam from 'react-webcam';
import './App.css'; // Ensure this CSS file is set up correctly

const WebcamCapture = () => {
  const webcamRef = useRef<Webcam>(null);

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      // Convert imageSrc (base64) to a blob and send it to the Flask server
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const formData = new FormData();
          formData.append('image', blob, 'screenshot.jpeg');
          return fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData,
          });
        })
        .then(response => response.text())
        .then(result => console.log(result))
        .catch(error => console.error('Error:', error));
    }
  }, [webcamRef]);

  return (
    <div className="webcam-container">
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        width="100%"
        videoConstraints={{ width: 1280, height: 720 }}
      />
      <button onClick={capture}>Capture photo</button>
    </div>
  );
};

export default WebcamCapture;
