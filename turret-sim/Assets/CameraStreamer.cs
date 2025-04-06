using UnityEngine;
using System.Collections;
using System.Net;
using System.Threading;
using System.IO;
using System.Text;

/// <summary>
/// CameraStreamer - Captures the main camera view and streams it for external analysis
/// Attach this script to your main camera in Unity
/// </summary>
public class CameraStreamer : MonoBehaviour
{
    [Header("Streaming Settings")]
    [Tooltip("Port to use for the HTTP server")]
    public int port = 8080;
    
    [Tooltip("Width of the streamed image")]
    public int streamWidth = 640;
    
    [Tooltip("Height of the streamed image")]
    public int streamHeight = 480;
    
    [Tooltip("Frames per second to capture")]
    public int frameRate = 30;
    
    [Tooltip("JPEG quality (1-100)")]
    [Range(1, 100)]
    public int jpegQuality = 75;
    
    [Tooltip("Enable streaming on start")]
    public bool autoStartStreaming = true;

    private HttpListener httpListener;
    private Thread serverThread;
    private bool isRunning = false;
    private Texture2D streamTexture;
    private RenderTexture cameraRenderTexture;
    private byte[] jpegBytes;
    private float frameTimer;

    // Initialization
    void Start()
    {
        // Initialize streaming texture
        streamTexture = new Texture2D(streamWidth, streamHeight, TextureFormat.RGB24, false);
        cameraRenderTexture = new RenderTexture(streamWidth, streamHeight, 24);
        jpegBytes = new byte[0];
        
        if (autoStartStreaming)
        {
            StartStreaming();
        }
    }

    // Clean up resources
    void OnDestroy()
    {
        StopStreaming();
        
        if (cameraRenderTexture != null)
        {
            cameraRenderTexture.Release();
            Destroy(cameraRenderTexture);
        }
        
        if (streamTexture != null)
        {
            Destroy(streamTexture);
        }
    }

    // Public method to start streaming
    public void StartStreaming()
    {
        if (isRunning) return;
        
        try
        {
            httpListener = new HttpListener();
            httpListener.Prefixes.Add($"http://*:{port}/");
            httpListener.Start();
            
            isRunning = true;
            serverThread = new Thread(ServerThreadMethod);
            serverThread.Start();
            
            Debug.Log($"Camera streaming started on port {port}");
            Debug.Log($"Access via http://localhost:{port}/camera or http://your-ip-address:{port}/camera");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to start streaming server: {e.Message}");
        }
    }

    // Public method to stop streaming
    public void StopStreaming()
    {
        isRunning = false;
        
        if (httpListener != null)
        {
            httpListener.Stop();
            httpListener.Close();
        }
        
        if (serverThread != null)
        {
            serverThread.Abort();
            serverThread = null;
        }
        
        Debug.Log("Camera streaming stopped");
    }

    // Capture and update the frame regularly
    void Update()
    {
        if (!isRunning) return;
        
        frameTimer += Time.deltaTime;
        
        // Capture frames at the specified frame rate
        if (frameTimer >= 1.0f / frameRate)
        {
            CaptureFrame();
            frameTimer = 0;
        }
    }

    // Capture the current camera view to a texture
    private void CaptureFrame()
    {
        Camera cam = GetComponent<Camera>();
        
        // Temporarily render the camera view to our render texture
        RenderTexture prevRT = cam.targetTexture;
        cam.targetTexture = cameraRenderTexture;
        cam.Render();
        cam.targetTexture = prevRT;
        
        // Read pixels from the render texture
        RenderTexture prevActive = RenderTexture.active;
        RenderTexture.active = cameraRenderTexture;
        streamTexture.ReadPixels(new Rect(0, 0, streamWidth, streamHeight), 0, 0);
        streamTexture.Apply();
        RenderTexture.active = prevActive;
        
        // Convert to JPEG
        jpegBytes = streamTexture.EncodeToJPG(jpegQuality);
    }

    // Server thread to handle HTTP requests
    private void ServerThreadMethod()
    {
        while (isRunning)
        {
            try
            {
                HttpListenerContext context = httpListener.GetContext();
                ProcessRequest(context);
            }
            catch (System.Exception e)
            {
                if (isRunning)
                {
                    Debug.LogError($"Error in server thread: {e.Message}");
                }
            }
        }
    }

    // Process incoming HTTP requests
    private void ProcessRequest(HttpListenerContext context)
    {
        string url = context.Request.Url.LocalPath;
        HttpListenerResponse response = context.Response;
        
        try
        {
            if (url == "/camera")
            {
                // Serve the MJPEG stream
                response.ContentType = "multipart/x-mixed-replace; boundary=--boundary";
                response.Headers.Add("Cache-Control", "no-cache, no-store, must-revalidate");
                response.Headers.Add("Pragma", "no-cache");
                response.Headers.Add("Expires", "0");
                
                using (Stream output = response.OutputStream)
                {
                    while (isRunning)
                    {
                        byte[] currentFrame = jpegBytes;
                        if (currentFrame.Length > 0)
                        {
                            try
                            {
                                // Write MJPEG header
                                string header = "\r\n--boundary\r\nContent-Type: image/jpeg\r\nContent-Length: " + currentFrame.Length + "\r\n\r\n";
                                byte[] headerBytes = Encoding.UTF8.GetBytes(header);
                                output.Write(headerBytes, 0, headerBytes.Length);
                                
                                // Write image data
                                output.Write(currentFrame, 0, currentFrame.Length);
                                output.Flush();
                                
                                // Limit frame rate on the server side as well
                                Thread.Sleep(1000 / frameRate);
                            }
                            catch
                            {
                                // Client probably disconnected
                                break;
                            }
                        }
                    }
                }
            }
            else if (url == "/snapshot")
            {
                // Serve a single JPEG image
                response.ContentType = "image/jpeg";
                response.ContentLength64 = jpegBytes.Length;
                response.OutputStream.Write(jpegBytes, 0, jpegBytes.Length);
                response.Close();
            }
            else if (url == "/")
            {
                // Serve a simple HTML page with links
                string html = @"
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Unity Camera Stream</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1 { color: #333; }
                        .container { display: flex; flex-direction: column; align-items: center; }
                        img { border: 1px solid #ddd; margin: 10px; max-width: 90%; }
                        a { display: block; margin: 10px; }
                    </style>
                </head>
                <body>
                    <div class='container'>
                        <h1>Unity Camera Stream</h1>
                        <img src='/camera' alt='Camera Stream' />
                        <a href='/snapshot' target='_blank'>Get Snapshot</a>
                    </div>
                </body>
                </html>";
                
                byte[] buffer = Encoding.UTF8.GetBytes(html);
                response.ContentType = "text/html";
                response.ContentLength64 = buffer.Length;
                response.OutputStream.Write(buffer, 0, buffer.Length);
                response.Close();
            }
            else
            {
                // Return 404 for unknown URLs
                response.StatusCode = 404;
                response.Close();
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error processing request: {e.Message}");
            try
            {
                response.StatusCode = 500;
                response.Close();
            }
            catch
            {
                // Ignore errors while sending error responses
            }
        }
    }
}

/// <summary>
/// RTSP alternative using a third-party library
/// </summary>
/* 
// NOTE: To use RTSP streaming, you'll need to:
// 1. Import a Unity-compatible RTSP server package (like "SimpleRtspServer" from the Asset Store)
// 2. Replace this commented code with the actual implementation based on the package you choose
// 3. Follow the package documentation for setup details

using UnityEngine;
using SimpleRtspServer; // Example namespace, will depend on the actual package you use

public class CameraRtspStreamer : MonoBehaviour
{
    [Header("RTSP Streaming Settings")]
    public string streamName = "camera";
    public int streamWidth = 640;
    public int streamHeight = 480;
    public int frameRate = 30;
    public int bitrate = 1000000; // 1 Mbps
    
    private RtspServer rtspServer;
    private RenderTexture cameraRenderTexture;
    private Texture2D streamTexture;
    private float frameTimer;

    void Start()
    {
        // Initialize textures
        cameraRenderTexture = new RenderTexture(streamWidth, streamHeight, 24);
        streamTexture = new Texture2D(streamWidth, streamHeight, TextureFormat.RGB24, false);
        
        // Initialize RTSP server
        rtspServer = new RtspServer();
        rtspServer.Initialize(streamName, streamWidth, streamHeight, frameRate, bitrate);
        rtspServer.Start();
        
        Debug.Log($"RTSP streaming started at rtsp://localhost:8554/{streamName}");
    }

    void Update()
    {
        frameTimer += Time.deltaTime;
        
        if (frameTimer >= 1.0f / frameRate)
        {
            CaptureAndSendFrame();
            frameTimer = 0;
        }
    }

    private void CaptureAndSendFrame()
    {
        Camera cam = GetComponent<Camera>();
        
        // Render camera to texture
        RenderTexture prevRT = cam.targetTexture;
        cam.targetTexture = cameraRenderTexture;
        cam.Render();
        cam.targetTexture = prevRT;
        
        // Read pixels
        RenderTexture prevActive = RenderTexture.active;
        RenderTexture.active = cameraRenderTexture;
        streamTexture.ReadPixels(new Rect(0, 0, streamWidth, streamHeight), 0, 0);
        streamTexture.Apply();
        RenderTexture.active = prevActive;
        
        // Send frame to RTSP server
        rtspServer.SendFrame(streamTexture.GetRawTextureData());
    }

    void OnDestroy()
    {
        if (rtspServer != null)
        {
            rtspServer.Stop();
        }
        
        if (cameraRenderTexture != null)
        {
            cameraRenderTexture.Release();
            Destroy(cameraRenderTexture);
        }
        
        if (streamTexture != null)
        {
            Destroy(streamTexture);
        }
    }
}
*/

// Python client code example for OpenCV
/*
# Python code for receiving and processing the stream
import cv2
import numpy as np

# For HTTP stream
def process_http_stream():
    # Connect to the HTTP stream
    cap = cv2.VideoCapture('http://localhost:8080/camera')
    
    if not cap.isOpened():
        print("Error: Could not connect to stream")
        return
        
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to receive frame")
            break
            
        # Process the frame with OpenCV
        # Example: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Example: Detect edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Display the original and processed frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Processed Frame', edges)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# For RTSP stream (when using the RTSP server implementation)
def process_rtsp_stream():
    # Connect to the RTSP stream
    cap = cv2.VideoCapture('rtsp://localhost:8554/camera')
    
    if not cap.isOpened():
        print("Error: Could not connect to RTSP stream")
        return
        
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to receive frame")
            break
            
        # Process with OpenCV
        # Your analysis code here
        
        cv2.imshow('RTSP Stream', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# Run the appropriate function
process_http_stream()  # For HTTP method
# process_rtsp_stream()  # For RTSP method when enabled
*/