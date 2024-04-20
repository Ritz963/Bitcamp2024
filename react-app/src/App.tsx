import React, { useEffect, useState } from 'react';
import io, { Socket } from 'socket.io-client';

// Define the URL of the Socket.IO server
const SERVER_URL = 'http://localhost:5000';

const App: React.FC = () => {
  const [frame, setFrame] = useState<string>('');

  useEffect(() => {
    const socket: Socket = io(SERVER_URL);

    // Setup event listener for receiving video frames
    socket.on('video_frame', (data: { data: string }) => {
      setFrame(data.data);
    });

    // Cleanup function to disconnect the socket when the component unmounts
    return () => {
      socket.off('video_frame');
      socket.disconnect();
    };
  }, []);

  return (
    <div className="App">
      {/* Display the video frame received from the server */}
      <img src={frame} alt="Video Feed" style={{ width: '100%', height: 'auto' }} />
    </div>
  );
};

export default App;
