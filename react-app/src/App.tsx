// App.tsx
import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';

const socket = io('http://127.0.0.1:5000');

function App() {
    const [frame, setFrame] = useState<string>('');

    useEffect(() => {
        socket.on('frame', (data: { data: string }) => {
            setFrame(data.data);
        });

        socket.emit('start_stream');

        return () => {
            socket.off('frame');
        };
    }, []);

    return (
        <div>
            <img src={`data:image/jpeg;base64,${frame}`} alt="Webcam Stream" />
        </div>
    );
}

export default App;
