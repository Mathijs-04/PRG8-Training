import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const enableWebcamButton = document.getElementById("webcamButton");
const exportButton = document.getElementById("exportButton");
const resetButton = document.getElementById("resetButton");

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");

const drawUtils = new DrawingUtils(canvasCtx);
let handLandmarker = undefined;
let webcamRunning = false;
let results = undefined;

let collectedData = { Schild: [], Magie: [], Zwaard: [] };
let keysPressed = {};

document.addEventListener("keydown", (event) => {
    keysPressed[event.key] = true;
});

document.addEventListener("keyup", (event) => {
    keysPressed[event.key] = false;
});

function isKeyPressed(key) {
    return keysPressed[key] === true;
}

const createHandLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm");
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 1
    });

    enableWebcamButton.addEventListener("click", enableCam);
    exportButton.addEventListener("click", exportTrainingData);
};

async function enableCam() {
    webcamRunning = true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        video.srcObject = stream;
        video.addEventListener("loadeddata", () => {
            canvasElement.style.width = video.videoWidth;
            canvasElement.style.height = video.videoHeight;
            canvasElement.width = video.videoWidth;
            canvasElement.height = video.videoHeight;
            document.querySelector(".videoView").style.height = video.videoHeight + "px";
            predictWebcam();
        });
    } catch (error) {
        console.error("Error accessing webcam:", error);
    }
}

async function predictWebcam() {
    results = await handLandmarker.detectForVideo(video, performance.now());

    if (results.landmarks.length > 0) {
        const hand = results.landmarks[0];
        const handData = hand.map((point) => [point.x, point.y, point.z]).flat();

        if (isKeyPressed("1")) {
            collectedData.Schild.push(handData);
        }

        if (isKeyPressed("2")) {
            collectedData.Magie.push(handData);
        }

        if (isKeyPressed("3")) {
            collectedData.Zwaard.push(handData);
        }

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        drawUtils.drawConnectors(hand, HandLandmarker.HAND_CONNECTIONS, { color: "#00FF00", lineWidth: 5 });
        drawUtils.drawLandmarks(hand, { radius: 4, color: "#FF0000", lineWidth: 2 });
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

if (navigator.mediaDevices?.getUserMedia) {
    createHandLandmarker();
}

function exportTrainingData() {
    const data = JSON.stringify(collectedData, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "TrainingData.json";
    a.click();
    URL.revokeObjectURL(url);
}

resetButton.addEventListener("click", () => {
    collectedData = { Schild: [], Magie: [], Zwaard: [] };
    console.log("Training data has been reset:", collectedData);
});
