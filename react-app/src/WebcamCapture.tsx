import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import './App.css';

const WebcamCapture = () => {
  const webcamRef = useRef<Webcam>(null);
  const [score, setScore] = useState<number | null>(null);  // State to store the score

  const capture = React.useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      fetch(imageSrc)
        .then(res => res.blob())
        .then(blob => {
          const formData = new FormData();
          formData.append('image', blob, 'screenshot.jpeg');
          return fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData,
          });
        })
        .then(response => response.json())  // Expecting a JSON response
        .then(data => {
          setScore(data.score);  // Set the score in state
          console.log('Score:', data.score);
        })
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
      {score !== null && <h2>Score: {score}</h2>}  {/* Display the score */}
    </div>
  );
};

export default WebcamCapture;
