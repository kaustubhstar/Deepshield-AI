export async function listVideoDevices(): Promise<MediaDeviceInfo[]> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((device) => device.kind === "videoinput");
}

export async function startStream(deviceId?: string, facingMode?: "user" | "environment") {
  const video: MediaTrackConstraints = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else if (facingMode) video.facingMode = { ideal: facingMode };
  return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

export function stopStream(stream: MediaStream | null) {
  stream?.getTracks().forEach((track) => track.stop());
}

export function captureJpeg(video: HTMLVideoElement, quality = 0.85): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas");
    if (!video.videoWidth || !video.videoHeight) {
      reject(new Error("The camera has not produced a frame yet. Wait a moment and try again."));
      return;
    }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      reject(new Error("Could not create a canvas context."));
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) reject(new Error("Frame capture failed."));
        else resolve(blob);
      },
      "image/jpeg",
      quality,
    );
  });
}

export function permissionMessage(error: unknown): string {
  if (error instanceof DOMException) {
    if (error.name === "NotAllowedError") return "Camera permission was denied. Allow camera access in the browser and try again.";
    if (error.name === "NotFoundError") return "No camera was found on this device.";
    if (error.name === "NotReadableError") return "The camera is already in use by another application.";
  }
  if (!window.isSecureContext) {
    return "Cameras require HTTPS (or localhost). Deploy the site with HTTPS for phone access.";
  }
  return "Could not start the camera in this browser.";
}
