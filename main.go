package main

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg" // Decoder
	_ "image/png" // If needed for icon or other assets
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"golang.org/x/image/font" // For FPS text overlay
	"golang.org/x/image/font/basicfont" // Basic font example
	"golang.org/x/image/math/fixed"
)

const (
	jpegMarkerSOI = "\xff\xd8" // Start Of Image
	jpegMarkerEOI = "\xff\xd9" // End Of Image
)

type DirectStreamingApp struct {
	fyneApp fyne.App
	window  fyne.Window

	// UI Elements
	hostEntry      *widget.Entry
	portEntry      *widget.Entry
	pathEntry      *widget.Entry
	streamType     *widget.Select
	performance    *widget.Select
	connectButton  *widget.Button
	showFPSCheck   *widget.Check
	reduceResCheck *widget.Check
	previewImage   *canvas.Image // Displays the video frame
	fpsLabel       *widget.Label // For overlay text
	noFeedLabel    *widget.Label // "No Camera Feed" text
	imageContainer *fyne.Container // Stack container for image + overlay
	logArea        *widget.Entry // MultiLineEntry for logs
	logScroller    *container.Scroll
	statusLabel    *widget.Label

	// State
	ctx           context.Context    // For signalling cancellation to goroutine
	cancelStream  context.CancelFunc // Function to call to cancel
	streamActive  bool               // Is the stream goroutine supposed to be running?
	connected     bool               // Is the connection established?
	frameChan     chan image.Image   // Channel to send decoded frames to UI thread
	logChan       chan string        // Channel for logging messages
	statusChan    chan string        // Channel for status updates
	currentFrame  image.Image        // Keep the last frame for drawing
	stateMutex    sync.RWMutex       // Protect access to shared state like fps counters, connected flag

	// FPS Calculation
	frameCount      int
	lastFPSTime     time.Time
	cameraFPS       float64
	uiFrameCount    int
	lastUIFPSTime   time.Time
	uiFPS           float64
}

func NewDirectStreamingApp() *DirectStreamingApp {
	a := app.New()
	// Consider setting a theme or icon
	// a.SetIcon(...)
	w := a.NewWindow("Go High-Speed Camera Preview")

	dsa := &DirectStreamingApp{
		fyneApp:   a,
		window:  w,
		frameChan: make(chan image.Image, 3), // Buffered channel
		logChan:   make(chan string, 50),    // Buffered log channel
		statusChan:make(chan string, 5),     // Buffered status channel
	}

	dsa.initUI()
	w.SetContent(dsa.createContent())
	w.Resize(fyne.NewSize(1000, 700))
	w.SetOnClosed(dsa.onClosing)

	// Start UI updater loop
	go dsa.uiUpdater()

	return dsa
}

func (d *DirectStreamingApp) initUI() {
	d.hostEntry = widget.NewEntry()
	d.hostEntry.SetText("127.0.0.1")
	d.portEntry = widget.NewEntry()
	d.portEntry.SetText("8081") // Common test port
	d.pathEntry = widget.NewEntry()
	d.pathEntry.SetText("/video")

	d.streamType = widget.NewSelect([]string{"MJPEG", "JPEG HTTP"}, func(s string) {})
	d.streamType.SetSelectedIndex(0)

	d.performance = widget.NewSelect([]string{"Maximum", "Balanced", "Quality"}, func(s string) {})
	d.performance.SetSelectedIndex(0)

	d.connectButton = widget.NewButton("Connect", d.toggleConnection)

	d.showFPSCheck = widget.NewCheck("Show FPS", func(b bool) {})
	d.showFPSCheck.SetChecked(true)
	d.reduceResCheck = widget.NewCheck("Reduce Resolution (Max Perf)", func(b bool) {})
	d.reduceResCheck.SetChecked(true) // Default to reducing for performance

	// Image display area
	d.previewImage = canvas.NewImageFromImage(nil) // Start empty
	d.previewImage.FillMode = canvas.ImageFillContain // Or ImageFillOriginal, ImageFillStretch
	d.previewImage.ScaleMode = canvas.ImageScaleFastest // Use fastest scaling initially

	// Overlay text (FPS and "No Feed")
	d.fpsLabel = widget.NewLabel("") // Initially empty
	d.fpsLabel.TextStyle = fyne.TextStyle{Bold: true}
	// Note: Fyne text color needs theme handling or custom widgets.
	// We can draw text onto the image instead for full color control.

	d.noFeedLabel = widget.NewLabel("No Camera Feed")
	d.noFeedLabel.Alignment = fyne.TextAlignCenter

	// Stack the image, "No Feed" text, and FPS text
	d.imageContainer = container.NewStack(
		d.noFeedLabel, // Initially visible
		d.previewImage, // Will be drawn over 'noFeedLabel' when frame arrives
		// container.NewVBox(d.fpsLabel), // Put FPS label in a container for positioning
        // Simpler approach for now: Draw FPS onto image
	)


	// Log area
	d.logArea = widget.NewMultiLineEntry()
	d.logArea.Disable() // Read-only
	d.logArea.Wrapping = fyne.TextWrapWord
	d.logScroller = container.NewScroll(d.logArea)
	// Consider setting a min size for the scroll container if needed

	// Status bar
	d.statusLabel = widget.NewLabel("Ready")
}

func (d *DirectStreamingApp) createContent() fyne.CanvasObject {
	// Left Panel (Controls)
	controls := container.NewVBox(
		widget.NewCard("Camera Connection", "", container.NewVBox(
			container.NewGridWithColumns(2,
				widget.NewLabel("Host:"), d.hostEntry,
				widget.NewLabel("Port:"), d.portEntry,
				widget.NewLabel("Path:"), d.pathEntry,
				widget.NewLabel("Stream Type:"), d.streamType,
				widget.NewLabel("Performance:"), d.performance,
			),
			d.connectButton,
			widget.NewSeparator(),
			d.showFPSCheck,
			d.reduceResCheck,
		)),
		// Add more control sections if needed
	)
	leftPanel := container.NewVBox(controls) // Add padding or structure if desired

	// Right Panel (Image) - Already created d.imageContainer

	// Bottom Panel (Log)
	logGroup := widget.NewCard("Log", "", d.logScroller)

	// Main Layout
	centerSplit := container.NewHSplit(leftPanel, d.imageContainer)
	centerSplit.SetOffset(0.3) // Adjust initial split ratio (30% left)

	content := container.NewBorder(
		nil,          // Top
		d.statusLabel, // Bottom
		nil,          // Left (handled by split)
		nil,          // Right (handled by split)
		centerSplit,  // Center
	)

	// Add the log area below the split
	mainLayout := container.NewVSplit(content, logGroup)
	mainLayout.SetOffset(0.8) // 80% for top content, 20% for logs


	return mainLayout
}

// log sends a message to the log channel for processing by uiUpdater
func (d *DirectStreamingApp) log(message string) {
	select {
	case d.logChan <- fmt.Sprintf("[%s] %s", time.Now().Format("15:04:05"), message):
	default:
		// Log channel full, discard message or log to stderr
		fmt.Println("Log channel full, discarding:", message)
	}
}

// status sends a message to the status channel
func (d *DirectStreamingApp) status(message string) {
	select {
	case d.statusChan <- message:
	default:
		fmt.Println("Status channel full, discarding:", message)
	}
}


// uiUpdater runs in a separate goroutine to update UI elements based on channel messages
func (d *DirectStreamingApp) uiUpdater() {
	ticker := time.NewTicker(30 * time.Millisecond) // Update UI roughly 30 times/sec
	defer ticker.Stop()

	var logBuffer strings.Builder // Buffer log messages for less frequent updates
	logUpdateTicker := time.NewTicker(250 * time.Millisecond)
	defer logUpdateTicker.Stop()

	for {
		select {
		case frame, ok := <-d.frameChan:
			if !ok { // Channel closed
				return
			}
			d.displayFrame(frame)
			d.stateMutex.Lock()
			d.uiFrameCount++
			d.stateMutex.Unlock()

		case logMsg, ok := <-d.logChan:
			if !ok {
				return
			}
			logBuffer.WriteString(logMsg + "\n")

		case statusMsg, ok := <-d.statusChan:
			if !ok {
				return
			}
			d.statusLabel.SetText(statusMsg)

		case <-logUpdateTicker.C:
			if logBuffer.Len() > 0 {
				// Append buffered logs
				currentLog := d.logArea.Text
				logBuffer.WriteString(currentLog) // Prepend new logs
				d.logArea.SetText(logBuffer.String())
				logBuffer.Reset()
				// Scroll to bottom (might need slight delay or better check)
				// d.logScroller.ScrollToBottom() // Fyne might auto-scroll, check behavior
			}

		case <-ticker.C:
			// Periodic updates (e.g., FPS calculation)
			d.updateFPS()
			// Potentially refresh widgets if needed, although SetText usually does
		}
	}
}

func (d *DirectStreamingApp) updateFPS() {
    d.stateMutex.Lock()
    defer d.stateMutex.Unlock()

    now := time.Now()
    // Camera FPS (updated by stream goroutine, just read here)
    // cameraFPS := d.cameraFPS

    // UI FPS
    uiDuration := now.Sub(d.lastUIFPSTime)
    if uiDuration >= 500*time.Millisecond && d.uiFrameCount > 0 {
        d.uiFPS = float64(d.uiFrameCount) / uiDuration.Seconds()
        d.lastUIFPSTime = now
        d.uiFrameCount = 0
    } else if uiDuration > 2*time.Second { // Reset if no frames for a while
		d.uiFPS = 0
		d.lastUIFPSTime = now
		d.uiFrameCount = 0
	}

    // Update status bar if connected
    if d.connected {
        statusText := fmt.Sprintf("Connected | Camera: %.1f FPS | UI: %.1f FPS", d.cameraFPS, d.uiFPS)
		d.status(statusText) // Send to channel for update
    }
}

// addOverlay draws FPS text onto the image
func addOverlay(img image.Image, camFPS, uiFPS float64, show bool) image.Image {
	if !show || img == nil {
		return img
	}

	text := fmt.Sprintf("Cam:%.1f UI:%.1f", camFPS, uiFPS)
	col := color.RGBA{Y: 255, A: 255} // Yellow text

	// Create a mutable copy of the image
	bounds := img.Bounds()
	newImg := image.NewRGBA(bounds)
	draw.Draw(newImg, bounds, img, bounds.Min, draw.Src)

	// Simple text drawing (adjust position as needed)
	// For better fonts, load a TTF file using golang.org/x/image/font/opentype
	point := fixed.Point26_6{X: fixed.Int26_6(10 * 64), Y: fixed.Int26_6(20 * 64)} // Position (x=10, y=20)

	d := &font.Drawer{
		Dst:  newImg,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13, // Basic font
		Dot:  point,
	}
	d.DrawString(text)

	return newImg
}


// displayFrame updates the image widget with a new frame
func (d *DirectStreamingApp) displayFrame(frame image.Image) {
	if frame == nil {
		return
	}

    d.stateMutex.RLock()
	showFPS := d.showFPSCheck.Checked
    camFPS := d.cameraFPS
    uiFPS := d.uiFPS
    d.stateMutex.RUnlock()

    // Add overlay text directly onto the image
    processedFrame := addOverlay(frame, camFPS, uiFPS, showFPS)

	// Update the image widget
	d.previewImage.Image = processedFrame
    d.previewImage.Refresh()

	// Hide "No Feed" label if it's the first frame
    if d.noFeedLabel.Visible() {
        d.noFeedLabel.Hide()
    }
    d.currentFrame = frame // Store unmodified frame if needed later
}

func (d *DirectStreamingApp) toggleConnection() {
	d.stateMutex.Lock()
	if d.streamActive {
		d.stateMutex.Unlock()
		d.disconnect("User disconnected")
	} else {
		d.stateMutex.Unlock()
		d.connect()
	}
}

func (d *DirectStreamingApp) connect() {
	host := d.hostEntry.Text
	portStr := d.portEntry.Text
	path := d.pathEntry.Text
	stype := d.streamType.Selected
	perfMode := d.performance.Selected
	reduceRes := d.reduceResCheck.Checked

	port, err := strconv.Atoi(portStr)
	if err != nil || port < 1 || port > 65535 {
		dialog.ShowError(fmt.Errorf("invalid port number: %s", portStr), d.window)
		return
	}

	// Basic validation
	if host == "" || path == "" {
		dialog.ShowError(fmt.Errorf("host and path cannot be empty"), d.window)
		return
	}

	d.log(fmt.Sprintf("Connecting to %s stream at http://%s:%d%s", stype, host, port, path))
	d.status("Connecting...")
	d.connectButton.SetText("Connecting...")
	d.connectButton.Disable()

    // Create context for cancellation
    d.stateMutex.Lock()
	d.ctx, d.cancelStream = context.WithCancel(context.Background())
    d.streamActive = true
    d.stateMutex.Unlock()

	// Launch the appropriate stream handler in a goroutine
	go func() {
		var err error
		if stype == "MJPEG" {
			err = d.mjpegStreamLoop(d.ctx, host, port, path, perfMode, reduceRes)
		} else {
			err = d.jpegHttpStreamLoop(d.ctx, host, port, path, perfMode, reduceRes)
		}

		// This block runs after the stream loop finishes (normally or with error)
        d.stateMutex.Lock()
		wasActive := d.streamActive // Check if we are still supposed to be active
		d.stateMutex.Unlock()

        if wasActive { // Only disconnect if we weren't already told to stop
			errMsg := "Stream ended unexpectedly"
			if err != nil {
				errMsg = fmt.Sprintf("Stream error: %v", err)
			}
            d.disconnect(errMsg) // Ensure cleanup and UI reset
        }
	}()
}

func (d *DirectStreamingApp) disconnect(reason string) {
	d.stateMutex.Lock()
	if !d.streamActive {
		d.stateMutex.Unlock()
		return // Already disconnected or disconnecting
	}

	// Signal the streaming goroutine to stop
	if d.cancelStream != nil {
		d.cancelStream()
        d.cancelStream = nil // Prevent double cancel
	}
	d.streamActive = false
    d.connected = false // Mark as not connected
	d.stateMutex.Unlock()

	d.log(fmt.Sprintf("Disconnected: %s", reason))
	d.status("Disconnected")
	d.connectButton.SetText("Connect")
	d.connectButton.Enable()

	// Clear display
	d.previewImage.Image = nil // Use nil image
	d.previewImage.Refresh()
    d.noFeedLabel.Show() // Show "No feed" text again

    d.stateMutex.Lock()
	d.cameraFPS = 0
	d.uiFPS = 0
    d.stateMutex.Unlock()

    // Drain frame channel to prevent blocking if producer stopped mid-send
    go func() {
        for range d.frameChan {}
    }()

}

func (d *DirectStreamingApp) onClosing() {
	d.log("Application closing...")
	d.disconnect("Window closed")
	// Potentially wait briefly for goroutines if needed, but context cancellation is usually enough
	// Close channels? Not strictly necessary if app exits, but good practice if module was reusable.
	// close(d.logChan)
	// close(d.statusChan)
	// close(d.frameChan) // Be careful closing if producer might still write
}

func (d *DirectStreamingApp) run() {
	d.window.ShowAndRun()
}

// --- Streaming Logic ---

func (d *DirectStreamingApp) mjpegStreamLoop(ctx context.Context, host string, port int, path string, perfMode string, reduceRes bool) error {
	url := fmt.Sprintf("http://%s:%d%s", host, port, path)
	d.log(fmt.Sprintf("MJPEG: Connecting to %s", url))

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "multipart/x-mixed-replace")
    req.Header.Set("Cache-Control", "no-cache")
    req.Header.Set("Connection", "keep-alive")


	client := &http.Client{Timeout: 10 * time.Second} // Timeout for initial connection
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("connection failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "multipart/x-mixed-replace") {
		// Maybe it's single JPEG? Or wrong endpoint?
		// Could try reading first few bytes for JPEG markers.
		return fmt.Errorf("unexpected content type: %s", contentType)
	}

	// Parse boundary
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || mediaType != "multipart/x-mixed-replace" || params["boundary"] == "" {
		return fmt.Errorf("invalid multipart boundary: %w", err)
	}
	boundary := params["boundary"]

	d.log("MJPEG: Connection established, reading stream...")
    d.stateMutex.Lock()
    d.connected = true
    d.lastFPSTime = time.Now()
	d.lastUIFPSTime = time.Now()
    d.frameCount = 0
    d.stateMutex.Unlock()

    d.status("Connected (MJPEG)") // Initial connected status
    d.connectButton.SetText("Disconnect")
    d.connectButton.Enable()


	// Use multipart reader
	mr := multipart.NewReader(resp.Body, boundary)
	jpegBytes := new(bytes.Buffer)

	for {
		select {
		case <-ctx.Done(): // Check for cancellation signal
			d.log("MJPEG: Cancellation received.")
			return ctx.Err() // Return context error (e.g., context.Canceled)
		default:
			// Proceed with reading
		}

		part, err := mr.NextPart()
		if err == io.EOF {
			d.log("MJPEG: End of stream (EOF)")
			return nil // Normal stream end
		}
		if err != nil {
			// Check context error again in case it happened during read
			if ctx.Err() != nil { return ctx.Err()}
			return fmt.Errorf("error reading part: %w", err)
		}

		// Check part headers (optional, e.g., for Content-Type: image/jpeg)
		// partContentType := part.Header.Get("Content-Type")
		// if !strings.Contains(partContentType, "image/jpeg") { ... }

		jpegBytes.Reset()
		_, err = io.Copy(jpegBytes, part)
		part.Close() // Important to close the part
		if err != nil {
			if ctx.Err() != nil { return ctx.Err()}
			d.log(fmt.Sprintf("MJPEG: Error copying part data: %v", err))
			continue // Skip this frame
		}

		// Decode JPEG
		img, err := jpeg.Decode(bytes.NewReader(jpegBytes.Bytes()))
		if err != nil {
			d.log(fmt.Sprintf("MJPEG: Error decoding JPEG: %v (bytes: %d)", err, jpegBytes.Len()))
			continue // Skip invalid frame
		}

		// --- Optional Resizing ---
		processedImg := img
		if reduceRes && (perfMode == "Maximum" || perfMode == "Balanced") {
			// Example: Halve resolution using NearestNeighbor for speed
			bounds := img.Bounds()
			newWidth := bounds.Dx() / 2
			newHeight := bounds.Dy() / 2
			if newWidth > 0 && newHeight > 0 {
				newRect := image.Rect(0, 0, newWidth, newHeight)
				resized := image.NewRGBA(newRect) // Or use original type if known
				// Use draw.NearestNeighbor for max speed, draw.ApproxBiLinear for balance
				scaler := draw.NearestNeighbor
				if perfMode == "Balanced" {
					 // scaler = draw.ApproxBiLinear // Needs golang.org/x/image/draw
					 // Note: draw package requires careful use, check documentation
					 // For simplicity, sticking to basic resize concept here.
					 // A library like `nfnt/resize` might be easier.
				}
				scaler.Scale(resized, newRect, img, bounds, draw.Over, nil)
				processedImg = resized
			}
		}
		// --- End Resizing ---


		// Send frame to UI thread (non-blocking)
		select {
		case d.frameChan <- processedImg:
            d.stateMutex.Lock()
            d.frameCount++
            now := time.Now()
            duration := now.Sub(d.lastFPSTime)
            if duration >= 500*time.Millisecond && d.frameCount > 0 {
                d.cameraFPS = float64(d.frameCount) / duration.Seconds()
                d.lastFPSTime = now
                d.frameCount = 0
            }
            d.stateMutex.Unlock()
		case <-ctx.Done():
			return ctx.Err()
		default:
			// Frame channel full, drop frame
			d.log("MJPEG: Frame channel full, dropping frame.")
		}
	}
}


func (d *DirectStreamingApp) jpegHttpStreamLoop(ctx context.Context, host string, port int, path string, perfMode string, reduceRes bool) error {
	baseURL := fmt.Sprintf("http://%s:%d%s", host, port, path)
	d.log(fmt.Sprintf("JPEG HTTP: Starting requests to %s", baseURL))

    // Initial connection check (optional but good)
    clientCheck := &http.Client{Timeout: 5 * time.Second}
    respCheck, errCheck := clientCheck.Get(baseURL)
    if errCheck != nil {
        return fmt.Errorf("initial connection check failed: %w", errCheck)
    }
    if respCheck.StatusCode != http.StatusOK {
         respCheck.Body.Close()
        return fmt.Errorf("initial check failed: status %s", respCheck.Status)
    }
    // Check content type?
     ct := respCheck.Header.Get("Content-Type")
     if !strings.Contains(ct, "image/jpeg") && !strings.Contains(ct, "image/jpg") {
         // Maybe it's actually MJPEG? Log a warning or error
          d.log(fmt.Sprintf("Warning: Expected JPEG Content-Type, got '%s'", ct))
     }
    respCheck.Body.Close()


	// Use a persistent client for subsequent requests
	client := &http.Client{Timeout: 3 * time.Second} // Shorter timeout per frame

	d.log("JPEG HTTP: Connection check OK.")
    d.stateMutex.Lock()
    d.connected = true
    d.lastFPSTime = time.Now()
	d.lastUIFPSTime = time.Now()
    d.frameCount = 0
    d.stateMutex.Unlock()

    d.status("Connected (JPEG HTTP)")
    d.connectButton.SetText("Disconnect")
    d.connectButton.Enable()

	targetInterval := time.Second / 60 // Target ~60 FPS requests max

	for {
        startTime := time.Now()

		select {
		case <-ctx.Done():
			d.log("JPEG HTTP: Cancellation received.")
			return ctx.Err()
		default:
			// Continue
		}

		req, err := http.NewRequestWithContext(ctx, "GET", baseURL, nil)
		if err != nil {
			d.log(fmt.Sprintf("JPEG HTTP: Failed to create request: %v", err))
			time.Sleep(500 * time.Millisecond) // Wait before retry
			continue
		}
        req.Header.Set("Cache-Control", "no-cache")
        req.Header.Set("Connection", "keep-alive") // Ask to keep alive


		resp, err := client.Do(req)
		if err != nil {
			if ctx.Err() != nil { return ctx.Err()} // Check context on error
			d.log(fmt.Sprintf("JPEG HTTP: Request failed: %v", err))
			time.Sleep(500 * time.Millisecond) // Wait before retry
			continue
		}

		if resp.StatusCode != http.StatusOK {
			d.log(fmt.Sprintf("JPEG HTTP: Bad status: %s", resp.Status))
			resp.Body.Close()
			time.Sleep(500 * time.Millisecond)
			continue
		}

		// Read the response body
		bodyBytes, err := io.ReadAll(resp.Body)
		resp.Body.Close() // Close body immediately
		if err != nil {
			if ctx.Err() != nil { return ctx.Err()}
			d.log(fmt.Sprintf("JPEG HTTP: Failed to read body: %v", err))
			continue
		}

		// Decode JPEG
		img, err := jpeg.Decode(bytes.NewReader(bodyBytes))
		if err != nil {
			d.log(fmt.Sprintf("JPEG HTTP: Error decoding JPEG: %v (bytes: %d)", err, len(bodyBytes)))
			continue
		}

		// --- Optional Resizing (same logic as MJPEG) ---
		processedImg := img
		if reduceRes && (perfMode == "Maximum" || perfMode == "Balanced") {
			bounds := img.Bounds()
			newWidth := bounds.Dx() / 2
			newHeight := bounds.Dy() / 2
			if newWidth > 0 && newHeight > 0 {
				newRect := image.Rect(0, 0, newWidth, newHeight)
				resized := image.NewRGBA(newRect)
				scaler := draw.NearestNeighbor
				if perfMode == "Balanced" {
					// scaler = draw.ApproxBiLinear
				}
				scaler.Scale(resized, newRect, img, bounds, draw.Over, nil)
				processedImg = resized
			}
		}
		// --- End Resizing ---

		// Send frame to UI thread
		select {
		case d.frameChan <- processedImg:
            d.stateMutex.Lock()
            d.frameCount++
            now := time.Now()
            duration := now.Sub(d.lastFPSTime)
            if duration >= 500*time.Millisecond && d.frameCount > 0 {
                d.cameraFPS = float64(d.frameCount) / duration.Seconds()
                d.lastFPSTime = now
                d.frameCount = 0
            }
            d.stateMutex.Unlock()
		case <-ctx.Done():
			return ctx.Err()
		default:
			d.log("JPEG HTTP: Frame channel full, dropping frame.")
		}

		// Control request rate
		elapsed := time.Since(startTime)
		waitTime := targetInterval - elapsed
		if waitTime > 0 {
			select {
			case <-time.After(waitTime):
			case <-ctx.Done():
				return ctx.Err()
			}
		}
	}
}


// --- Main Function ---
func main() {
	dsa := NewDirectStreamingApp()
	dsa.run()
}