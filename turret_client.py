import socket
import threading


class TurretClient:
    def __init__(self, host='127.0.0.1', port=8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        self.receive_thread = None
        self.lock = threading.Lock()
        self.response_callback = None
        self.error_callback = None
        # Track current position
        self.current_yaw = 0
        self.current_pitch = 0
        
    def connect(self):
        """Establish connection to the turret controller"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """Disconnect from the turret controller"""
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
    
    def _receive_loop(self):
        """Thread function to receive responses from the server"""
        buffer = ""
        
        while self.connected:
            try:
                data = self.socket.recv(1024).decode('ascii')
                if not data:
                    # Connection closed
                    self.connected = False
                    if self.error_callback:
                        self.error_callback("Connection closed by server")
                    break
                
                buffer += data
                
                # Process complete messages
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if self.response_callback:
                        self.response_callback(line)
            except Exception as e:
                if self.connected and self.error_callback:
                    self.error_callback(f"Receive error: {str(e)}")
                self.connected = False
                break
    
    def send_command(self, command):
        """Send a command to the turret controller"""
        if not self.connected:
            if self.error_callback:
                self.error_callback("Not connected")
            return False
        
        try:
            with self.lock:
                self.socket.sendall(f"{command}\n".encode('ascii'))
            return True
        except Exception as e:
            self.connected = False
            if self.error_callback:
                self.error_callback(f"Send error: {str(e)}")
            return False
    
    def laser_on(self):
        """Turn on the laser"""
        return self.send_command("LASER:ON")
    
    def laser_off(self):
        """Turn off the laser"""
        return self.send_command("LASER:OFF")
    
    def rotate(self, yaw, pitch):
        """Rotate the turret"""
        success = self.send_command(f"ROTATE:{yaw},{pitch}")
        if success:
            self.current_yaw = yaw
            self.current_pitch = pitch
        return success
    
    def ping(self):
        """Ping the server to check if it's alive"""
        return self.send_command("PING")