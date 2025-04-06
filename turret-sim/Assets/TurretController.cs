using UnityEngine;
using System.Collections;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System;
using System.Collections.Generic;

public class SimpleTurretController : MonoBehaviour
{
    [Header("Turret Components")]
    [SerializeField] private Transform turretBase;
    [SerializeField] private Transform barrelPivot;

    [Header("Custom Laser Origin")]
    [SerializeField] private Transform customLaserOrigin;
    [SerializeField] private bool useCustomLaserOrigin = false;

    [Header("Rotation Settings")]
    [SerializeField] private float yawSpeed = 120f;
    [SerializeField] private float pitchSpeed = 90f;
    [SerializeField] private float minPitch = -40f;
    [SerializeField] private float maxPitch = 60f;

    [Header("Laser Settings")]
    [SerializeField] private bool laserEnabled = true;
    [SerializeField] private GameObject laserDot;
    [SerializeField] private GameObject laserBeam;
    [SerializeField] private float laserMaxDistance = 100f;
    [SerializeField] private bool invertLaserDirection = true;
    [SerializeField] private bool hideBeamKeepDot = false;

    [Header("Network Settings")]
    [SerializeField] private int networkPort = 8888;
    [SerializeField] private bool startServerOnAwake = true;

    // Current rotation values
    private float currentYaw = 0f;
    private float currentPitch = 0f;

    // Components
    private LineRenderer lineRenderer;
    private GameObject dotInstance;

    // TCP server variables
    private TcpListener tcpListener;
    private Thread listenerThread;
    private bool isRunning = false;
    private Queue<Action> mainThreadQueue = new Queue<Action>();
    private object lockObject = new object();

    private void Awake()
    {
        // Create laser components if they don't exist
        if (laserBeam == null)
        {
            laserBeam = new GameObject("LaserBeam");
            laserBeam.transform.parent = transform;
            lineRenderer = laserBeam.AddComponent<LineRenderer>();
            lineRenderer.startWidth = 0.05f;
            lineRenderer.endWidth = 0.05f;
            lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
            lineRenderer.startColor = Color.red;
            lineRenderer.endColor = Color.red;
            lineRenderer.positionCount = 2;
            lineRenderer.enabled = false;
        }
        else
        {
            lineRenderer = laserBeam.GetComponent<LineRenderer>();
            if (lineRenderer == null)
            {
                lineRenderer = laserBeam.AddComponent<LineRenderer>();
                lineRenderer.startWidth = 0.05f;
                lineRenderer.endWidth = 0.05f;
                lineRenderer.material = new Material(Shader.Find("Sprites/Default"));
                lineRenderer.startColor = Color.red;
                lineRenderer.endColor = Color.red;
                lineRenderer.positionCount = 2;
                lineRenderer.enabled = false;
            }
        }

        if (laserDot == null)
        {
            laserDot = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            laserDot.name = "LaserDot";
            laserDot.transform.localScale = new Vector3(0.2f, 0.2f, 0.2f);
            Destroy(laserDot.GetComponent<Collider>());
            laserDot.GetComponent<Renderer>().material.color = Color.red;
            laserDot.SetActive(false);
        }

        dotInstance = Instantiate(laserDot);
        dotInstance.SetActive(false);
    }

    private void Start()
    {
        Debug.Log("SimpleTurretController starting");

        // Verify components
        if (turretBase == null)
        {
            turretBase = transform;
            Debug.Log("Using self as turretBase");
        }

        if (barrelPivot == null && !useCustomLaserOrigin)
        {
            barrelPivot = transform.Find("BarelPivot");
            if (barrelPivot == null)
            {
                Debug.LogError("BarelPivot not found! Turret aiming won't work properly.");
            }
            else
            {
                Debug.Log("Found BarelPivot reference");
            }
        }

        // Store initial rotation
        if (turretBase != null)
            currentYaw = turretBase.localEulerAngles.y;
        if (barrelPivot != null)
            currentPitch = barrelPivot.localEulerAngles.x;
        if (currentPitch > 180f)
            currentPitch -= 360f;

        // Start network server
        if (startServerOnAwake)
        {
            StartNetworkServer();
        }

        Debug.Log("SimpleTurretController started");
    }

    private void Update()
    {
        // Process network commands on main thread
        ProcessMainThreadQueue();

        // Handle keyboard input
        HandleInput();

        // Debug: Press F5 to fire
        if (Input.GetKeyDown(KeyCode.F5))
        {
            Debug.Log("F5 pressed - firing laser");
            FireLaser();
        }
    }

    private void HandleInput()
    {
        float yawInput = Input.GetAxis("Horizontal");
        float pitchInput = Input.GetAxis("Vertical");

        if (Mathf.Abs(yawInput) > 0.001f || Mathf.Abs(pitchInput) > 0.001f)
        {
            RotateTurret(yawInput, pitchInput);
        }

        if (Input.GetButtonDown("Fire1"))
        {
            FireLaser();
        }
    }

    private void OnDestroy()
    {
        StopNetworkServer();
    }

    #region Turret Control

    public void RotateTurret(float yawInput, float pitchInput)
    {
        if (turretBase == null || barrelPivot == null)
            return;

        // Update yaw (base rotation)
        currentYaw += yawInput * yawSpeed * Time.deltaTime;
        turretBase.localRotation = Quaternion.Euler(0f, currentYaw, 0f);

        // Update pitch (barrel angle)
        currentPitch -= pitchInput * pitchSpeed * Time.deltaTime;
        currentPitch = Mathf.Clamp(currentPitch, minPitch, maxPitch);
        barrelPivot.localRotation = Quaternion.Euler(currentPitch, 0f, 0f);
    }

    public void SetTurretRotation(float yaw, float pitch)
    {
        if (turretBase == null || barrelPivot == null)
        {
            Debug.LogError("Cannot set rotation: turretBase or barrelPivot is null");
            return;
        }

        // Convert from -1,1 range to actual angles
        float targetYaw = yaw * 180f;  // -1 to 1 maps to -180 to 180 degrees
        float targetPitch = pitch * 40f;  // -1 to 1 maps to -40 to 40 degrees

        // Apply limits
        targetPitch = Mathf.Clamp(targetPitch, minPitch, maxPitch);

        // Set rotation directly
        turretBase.localRotation = Quaternion.Euler(0f, targetYaw, 0f);
        barrelPivot.localRotation = Quaternion.Euler(targetPitch, 0f, 0f);

        // Update current values
        currentYaw = targetYaw;
        currentPitch = targetPitch;

        Debug.Log($"Set turret rotation: yaw={targetYaw}, pitch={targetPitch}");
    }

    public void FireLaser()
    {
        if (!laserEnabled || (barrelPivot == null && (!useCustomLaserOrigin || customLaserOrigin == null)))
        {
            Debug.Log("Laser is disabled or required transform is null");
            return;
        }

        Debug.Log("Firing laser");

        // Get firing position and direction
        Vector3 firePos;
        Vector3 fireDir;
        
        if (useCustomLaserOrigin && customLaserOrigin != null)
        {
            // Use custom transform
            firePos = customLaserOrigin.position;
            fireDir = invertLaserDirection ? -customLaserOrigin.forward : customLaserOrigin.forward;
            Debug.Log($"Using custom laser origin at {firePos}");
        }
        else
        {
            // Use default barrel pivot
            firePos = barrelPivot.position;
            fireDir = invertLaserDirection ? -barrelPivot.forward : barrelPivot.forward;
        }

        // Draw debug ray to verify direction
        Debug.DrawRay(firePos, fireDir * 10f, Color.yellow, 1.0f);

        // Perform raycast
        RaycastHit hit;
        bool didHit = Physics.Raycast(firePos, fireDir, out hit, laserMaxDistance);

        // Calculate end position
        Vector3 hitPos = didHit ? hit.point : firePos + fireDir * laserMaxDistance;

        // Show visual laser
        ShowLaser(firePos, hitPos);

        if (didHit)
        {
            Debug.Log($"Laser hit: {hit.collider.gameObject.name} at {hit.point}");
        }
        else
        {
            Debug.Log("Laser did not hit anything");
        }
    }

    private void ShowLaser(Vector3 startPos, Vector3 endPos)
    {
        // Update line renderer only if beam is not hidden
        if (lineRenderer != null && !hideBeamKeepDot)
        {
            lineRenderer.enabled = true;
            lineRenderer.SetPosition(0, startPos);
            lineRenderer.SetPosition(1, endPos);
        }

        // Update dot position (always shown)
        if (dotInstance != null)
        {
            dotInstance.SetActive(true);
            dotInstance.transform.position = endPos;
        }

        // Start coroutine to disable laser after delay
        // StartCoroutine(HideLaserAfterDelay(0.5f));
    }

    private IEnumerator HideLaserAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);

        // Always hide the line renderer after delay
        if (lineRenderer != null)
            lineRenderer.enabled = false;

        // Always hide the dot after delay
        if (dotInstance != null)
            dotInstance.SetActive(false);
    }

    public void SetLaserEnabled(bool enabled)
    {
        laserEnabled = enabled;
        Debug.Log($"Laser enabled: {enabled}");

        if (!enabled)
        {
            if (lineRenderer != null)
                lineRenderer.enabled = false;

            if (dotInstance != null)
                dotInstance.SetActive(false);
        }
    }
    
    public void SetHideBeamKeepDot(bool hide)
    {
        hideBeamKeepDot = hide;
        Debug.Log($"Hide beam, keep dot: {hide}");
        
        // If we're hiding the beam immediately, disable the line renderer
        if (hide && lineRenderer != null)
        {
            lineRenderer.enabled = false;
        }
    }
    
    // Set custom laser origin transform
    public void SetCustomLaserOrigin(Transform newOrigin, bool useIt = true)
    {
        customLaserOrigin = newOrigin;
        useCustomLaserOrigin = useIt;
        Debug.Log($"Custom laser origin set to {(newOrigin != null ? newOrigin.name : "null")}. Using custom origin: {useIt}");
    }

    #endregion

    #region Network Control

    private void StartNetworkServer()
    {
        try
        {
            tcpListener = new TcpListener(IPAddress.Any, networkPort);
            listenerThread = new Thread(new ThreadStart(ListenForClients));
            listenerThread.IsBackground = true;
            isRunning = true;
            listenerThread.Start();
            Debug.Log($"Network server started on port {networkPort}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error starting server: {e.Message}");
        }
    }

    private void StopNetworkServer()
    {
        isRunning = false;
        if (tcpListener != null)
        {
            tcpListener.Stop();
        }
    }

    private void ListenForClients()
    {
        tcpListener.Start();

        while (isRunning)
        {
            try
            {
                TcpClient client = tcpListener.AcceptTcpClient();
                ThreadPool.QueueUserWorkItem(HandleClientConnection, client);
            }
            catch (SocketException)
            {
                // Expected when stopping server
                if (isRunning) Debug.LogError("Socket exception in listener");
            }
            catch (Exception e)
            {
                if (isRunning) Debug.LogError($"Error in listener: {e.Message}");
            }
        }
    }

    private void HandleClientConnection(object clientObj)
    {
        TcpClient client = (TcpClient)clientObj;
        NetworkStream stream = client.GetStream();
        string clientEndPoint = client.Client.RemoteEndPoint.ToString();

        Debug.Log($"Client connected: {clientEndPoint}");
        SendResponse(stream, "CONNECTED:OK");

        byte[] buffer = new byte[1024];
        StringBuilder messageBuilder = new StringBuilder();

        while (isRunning && client.Connected)
        {
            messageBuilder.Clear();
            int bytesRead = 0;

            try
            {
                if (!stream.DataAvailable)
                {
                    Thread.Sleep(10);
                    continue;
                }

                bytesRead = stream.Read(buffer, 0, buffer.Length);
                if (bytesRead == 0) break;

                string message = Encoding.ASCII.GetString(buffer, 0, bytesRead);
                messageBuilder.Append(message);

                string completeMessage = messageBuilder.ToString();
                if (completeMessage.Contains("\n"))
                {
                    string[] messages = completeMessage.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
                    foreach (string msg in messages)
                    {
                        ProcessCommand(msg.Trim(), stream);
                    }
                    messageBuilder.Clear();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"Error handling client: {e.Message}");
                break;
            }
        }

        client.Close();
        Debug.Log($"Client disconnected: {clientEndPoint}");
    }

    private void ProcessCommand(string command, NetworkStream stream)
    {
        Debug.Log($"Received command: {command}");

        try
        {
            string[] parts = command.Split(new[] { ':' }, 2);
            string cmd = parts[0].ToUpperInvariant();
            string param = parts.Length > 1 ? parts[1] : string.Empty;

            switch (cmd)
            {
                case "PING":
                    SendResponse(stream, "PONG");
                    break;

                case "LASER":
                    ProcessLaserCommand(param, stream);
                    break;

                case "ROTATE":
                    ProcessRotateCommand(param, stream);
                    break;

                case "FIRE":
                    EnqueueOnMainThread(() => FireLaser());
                    SendResponse(stream, "FIRE:OK");
                    break;

                case "SETORIGIN":
                    ProcessSetOriginCommand(param, stream);
                    break;
                    
                case "HIDEBEAM":
                    ProcessHideBeamCommand(param, stream);
                    break;

                default:
                    SendResponse(stream, $"ERROR:Unknown command '{cmd}'");
                    break;
            }
        }
        catch (Exception e)
        {
            SendResponse(stream, $"ERROR:{e.Message}");
        }
    }

    private void ProcessLaserCommand(string param, NetworkStream stream)
    {
        param = param.ToUpperInvariant();
        if (param == "ON")
        {
            EnqueueOnMainThread(() => {
                SetLaserEnabled(true);
                FireLaser();
            });
            SendResponse(stream, "LASER:ON");
        }
        else if (param == "OFF")
        {
            EnqueueOnMainThread(() => SetLaserEnabled(false));
            SendResponse(stream, "LASER:OFF");
        }
        else
        {
            SendResponse(stream, $"ERROR:Unknown laser parameter '{param}'");
        }
    }

    private void ProcessRotateCommand(string param, NetworkStream stream)
    {
        try
        {
            string[] values = param.Split(',');
            if (values.Length != 2)
            {
                SendResponse(stream, "ERROR:Rotation requires 2 parameters (yaw,pitch)");
                return;
            }

            float yaw = float.Parse(values[0]);
            float pitch = float.Parse(values[1]);

            EnqueueOnMainThread(() => SetTurretRotation(yaw, pitch));
            SendResponse(stream, $"ROTATE:{yaw},{pitch}");
        }
        catch (FormatException)
        {
            SendResponse(stream, "ERROR:Invalid rotation parameters");
        }
        catch (Exception e)
        {
            SendResponse(stream, $"ERROR:{e.Message}");
        }
    }

    private void ProcessSetOriginCommand(string param, NetworkStream stream)
    {
        try
        {
            string[] values = param.Split(',');
            if (values.Length != 1 && values.Length != 2)
            {
                SendResponse(stream, "ERROR:SETORIGIN requires 1 or 2 parameters (transformPath,useCustom)");
                return;
            }

            string transformPath = values[0];
            bool useCustom = values.Length == 2 ? bool.Parse(values[1]) : true;

            EnqueueOnMainThread(() => {
                Transform newOrigin = GameObject.Find(transformPath)?.transform;
                if (newOrigin != null)
                {
                    SetCustomLaserOrigin(newOrigin, useCustom);
                    SendResponse(stream, $"SETORIGIN:OK,{transformPath},{useCustom}");
                }
                else
                {
                    SendResponse(stream, $"ERROR:Transform not found at path '{transformPath}'");
                }
            });
        }
        catch (FormatException)
        {
            SendResponse(stream, "ERROR:Invalid SETORIGIN parameters");
        }
        catch (Exception e)
        {
            SendResponse(stream, $"ERROR:{e.Message}");
        }
    }
    
    private void ProcessHideBeamCommand(string param, NetworkStream stream)
    {
        try
        {
            bool hideBeam = bool.Parse(param);
            
            EnqueueOnMainThread(() => {
                SetHideBeamKeepDot(hideBeam);
            });
            
            SendResponse(stream, $"HIDEBEAM:{hideBeam}");
        }
        catch (FormatException)
        {
            SendResponse(stream, "ERROR:Invalid HIDEBEAM parameter. Expected true or false");
        }
        catch (Exception e)
        {
            SendResponse(stream, $"ERROR:{e.Message}");
        }
    }

    private void SendResponse(NetworkStream stream, string response)
    {
        try
        {
            Debug.Log($"Sending response: {response}");
            response += "\n";
            byte[] data = Encoding.ASCII.GetBytes(response);
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending response: {e.Message}");
        }
    }

    private void EnqueueOnMainThread(Action action)
    {
        lock (lockObject)
        {
            mainThreadQueue.Enqueue(action);
        }
    }

    private void ProcessMainThreadQueue()
    {
        if (mainThreadQueue.Count > 0)
        {
            lock (lockObject)
            {
                while (mainThreadQueue.Count > 0)
                {
                    Action action = mainThreadQueue.Dequeue();
                    try
                    {
                        action();
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"Error executing action: {e.Message}");
                    }
                }
            }
        }
    }

    #endregion
}