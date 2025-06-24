import socket
import threading


class TurretClient:
    """
    Network client for communicating with a turret controller.
    
    This class provides a TCP socket interface to send commands to a turret
    controller and receive responses. It handles connection management,
    command sending, and response processing in a thread-safe manner.
    
    The protocol is text-based with commands like:
    - LASER:ON / LASER:OFF
    - ROTATE:yaw,pitch
    - PING
    """
    
    def __init__(self, host='127.0.0.1', port=8888):
        """
        Initialize turret client with connection parameters.
        
        Args:
            host: Turret controller hostname or IP address
            port: Turret controller port number
        """
        self.host = host
        self.port = port
        self.socket = None              # TCP socket for communication
        self.connected = False          # Connection status flag
        self.receive_thread = None      # Thread for receiving responses
        self.lock = threading.Lock()    # Thread synchronization lock
        
        # Callback functions for handling events
        self.response_callback = None   # Called when response received
        self.error_callback = None      # Called when error occurs
        
        # Track current turret position
        self.current_yaw = 0    # Current yaw/pan angle
        self.current_pitch = 0  # Current pitch/tilt angle
        
    def connect(self):
        """
        Establish TCP connection to the turret controller.
        
        This method creates a socket connection and starts a background
        thread to receive responses from the server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create TCP socket and connect
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            
            # Start background thread to receive responses
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.receive_thread.daemon = True  # Thread will exit when main program exits
            self.receive_thread.start()
            
            return True
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Connection error: {str(e)}")
            return False
    
    def disconnect(self):
        """
        Disconnect from the turret controller and cleanup resources.
        """
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass  # Ignore errors during cleanup
            self.socket = None
    
    def _receive_loop(self):
        """
        Background thread function to continuously receive responses from server.
        
        This method runs in a separate thread and processes incoming data
        from the turret controller. It handles partial messages and
        calls the response callback for complete messages.
        """
        buffer = ""  # Buffer for partial messages
        
        while self.connected:
            try:
                # Receive data from socket
                data = self.socket.recv(1024).decode('ascii')
                if not data:
                    # Connection closed by server
                    self.connected = False
                    if self.error_callback:
                        self.error_callback("Connection closed by server")
                    break
                
                # Add received data to buffer
                buffer += data
                
                # Process complete messages (terminated by newline)
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
        """
        Send a command to the turret controller.
        
        Commands are sent as ASCII text terminated by newline.
        This method is thread-safe and can be called from multiple threads.
        
        Args:
            command: Command string to send (without newline)
            
        Returns:
            bool: True if command sent successfully, False otherwise
        """
        if not self.connected:
            if self.error_callback:
                self.error_callback("Not connected")
            return False
        
        try:
            # Use lock to ensure thread-safe sending
            with self.lock:
                self.socket.sendall(f"{command}\n".encode('ascii'))
            return True
        except Exception as e:
            self.connected = False
            if self.error_callback:
                self.error_callback(f"Send error: {str(e)}")
            return False
    
    def laser_on(self):
        """
        Turn on the laser.
        
        Returns:
            bool: True if command sent successfully
        """
        return self.send_command("LASER:ON")
    
    def laser_off(self):
        """
        Turn off the laser.
        
        Returns:
            bool: True if command sent successfully
        """
        return self.send_command("LASER:OFF")
    
    def rotate(self, yaw, pitch):
        """
        Rotate the turret to specified angles.
        
        Args:
            yaw: Horizontal rotation angle (pan)
            pitch: Vertical rotation angle (tilt)
            
        Returns:
            bool: True if command sent successfully
        """
        success = self.send_command(f"ROTATE:{yaw},{pitch}")
        if success:
            # Update tracked position if command was sent successfully
            self.current_yaw = yaw
            self.current_pitch = pitch
        return success
    
    def ping(self):
        """
        Send a ping command to check if server is alive.
        
        Returns:
            bool: True if command sent successfully
        """
        return self.send_command("PING")