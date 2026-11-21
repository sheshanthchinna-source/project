let stream = null;
let predictionInterval = null;
let hands = null;
let detectionBusy = false;
let predictionBusy = false;
let predictionState = null;
let sequence = [];
let noHandCount = 0;
let frameCounter = 0;
const SEQUENCE_LENGTH = 30;
const MAX_NO_HAND_FRAMES = 3;
const PREDICT_EVERY = 3;

let detectedWord = "Unknown";
let detectedAlphabet = "Unknown";

// ================= SENTENCE =================
let sentence = "";

// ================= ALPHABET SETTINGS =================
let lastAddedAlphabet = "";
let alphabetStableCount = 0;
let lastAlphabetAddTime = 0;

const ALPHABET_CONFIRM_FRAMES = 5;
const ALPHABET_COOLDOWN = 1200;

// ================= WORD SETTINGS =================
let lastDetectedWord = "";
let wordStableCount = 0;
let lastAddedWord = "";
let lastWordAddTime = 0;

const WORD_CONFIRM_FRAMES = 5;
const WORD_COOLDOWN = 1800;


// ================= CAMERA =================
const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
function getMode() {
    return document.querySelector(
        'input[name="detectMode"]:checked'
    )?.value || "both";
}


// ================= START CAMERA =================
async function startCamera() {
    try {
        if (stream) return;

        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;

        document.getElementById("cameraMessage").innerText =
            "Camera started - Show your hand";

        video.onloadedmetadata = () => {
            if (predictionInterval) {
                clearInterval(predictionInterval);
            }
            predictionInterval = setInterval(processVideoFrame, 100);
        };

    } catch (error) {

        console.error("Camera error:", error);

        alert(
            "Unable to access camera. Please allow camera permission."
        );
    }
}


async function processVideoFrame() {
    if (!stream || detectionBusy || video.readyState !== 4) return;
    try {
        if (!hands) {
            hands = new Hands({
                locateFile: file =>
                    `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
            });
            hands.setOptions({
                maxNumHands: 1,
                modelComplexity: 0,
                minDetectionConfidence: 0.6,
                minTrackingConfidence: 0.6
            });
            hands.onResults(handleHandResults);
        }

        detectionBusy = true;
        await hands.send({ image: video });
    } catch (error) {
        console.error("Hand detection error:", error);
    } finally {
        detectionBusy = false;
    }
}

function handleHandResults(results) {
    if (!stream) return;

    const detectedHands = results.multiHandLandmarks;
    if (!detectedHands || detectedHands.length === 0) {
        noHandCount++;
        if (predictionState) {
            predictionState.word_history = [];
            predictionState.alphabet_history = [];
            predictionState.word_score_history = [];
            predictionState.alphabet_score_history = [];
        }
        showHandStatus(false);
        document.getElementById("word").innerText = "No hand detected";
        setConfidence(0);

        if (noHandCount >= MAX_NO_HAND_FRAMES) {
            sequence = [];
            predictionState = null;
            frameCounter = 0;
        }
        updateFrameProgress();
        return;
    }

    noHandCount = 0;
    showHandStatus(true);
    sequence.push(detectedHands[0].flatMap(point => [point.x, point.y, point.z]));
    if (sequence.length > SEQUENCE_LENGTH) sequence.shift();
    updateFrameProgress();

    if (sequence.length < SEQUENCE_LENGTH) {
        document.getElementById("word").innerText =
            `Collecting movement... (${sequence.length}/${SEQUENCE_LENGTH})`;
        setConfidence(0);
        return;
    }

    frameCounter++;
    if (frameCounter % PREDICT_EVERY === 0) {
        sendSequence();
    } else if (detectedWord === "Unknown") {
        document.getElementById("word").innerText = "Analyzing movement...";
    }
}

function updateFrameProgress() {
    const progress = document.getElementById("frameProgress");
    if (progress) progress.innerText = `${sequence.length} / ${SEQUENCE_LENGTH} frames`;
}

function showHandStatus(detected) {
    const handStatus = document.getElementById("handStatus");
    if (handStatus) {
        handStatus.innerText = detected ? "🟢 Hand detected" : "🔴 No hand detected";
    }
}

function setConfidence(confidence) {
    document.getElementById("confidence").innerText = `${confidence.toFixed(1)}%`;
    document.getElementById("confidenceBar").style.width =
        `${Math.min(confidence, 100)}%`;
}

// ================= SEND LANDMARK SEQUENCE =================
async function sendSequence() {
    if (predictionBusy) return;
    predictionBusy = true;
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                sequence,
                state: predictionState
            })
        });
        const result = await response.json();
        if (!response.ok || result.error) {
            console.error("Prediction error:", result.error || response.statusText);
            return;
        }
        if (result.state) predictionState = result.state;
        updatePrediction(result);
    } catch (error) {
        console.error("Prediction error:", error);
    } finally {
        predictionBusy = false;
    }
}

function updatePrediction(result) {
    detectedWord = result.word || "Unknown";
    detectedAlphabet = result.alphabet || "Unknown";

    const mode = getMode();
    let displayPrediction = "Waiting...";
    if (mode === "alphabet") {
        displayPrediction = detectedAlphabet;
    } else if (mode === "word") {
        displayPrediction = detectedWord;
    } else if (detectedWord !== "Unknown" && detectedWord !== "") {
        displayPrediction = detectedWord;
    } else {
        displayPrediction = detectedAlphabet;
    }
    document.getElementById("word").innerText = displayPrediction;

    const confidence = mode === "alphabet"
        ? (result.alphabet_confidence || 0) * 100
        : (result.confidence || 0) * 100;
    setConfidence(confidence);
    showHandStatus(result.hand_detected);

    if (mode === "alphabet") {
        processAlphabet(detectedAlphabet, result.alphabet_confidence || 0);
    } else if (mode === "word") {
        processWord(detectedWord, result.confidence || 0);
    } else if (detectedWord && detectedWord !== "Unknown") {
        processWord(detectedWord, result.confidence || 0);
    } else {
        processAlphabet(detectedAlphabet, result.alphabet_confidence || 0);
    }
}


// ================= ALPHABET PROCESSING =================

function processAlphabet(letter, confidence) {

    if (
        !/^[A-Z]$/.test(letter) ||
        confidence < 0.75
    ) {

        alphabetStableCount = 0;
        return;
    }


    if (letter === lastAddedAlphabet) {

        alphabetStableCount++;

    } else {

        lastAddedAlphabet = letter;
        alphabetStableCount = 1;
    }


    const now = Date.now();


    if (
        alphabetStableCount >=
        ALPHABET_CONFIRM_FRAMES &&

        now - lastAlphabetAddTime >=
        ALPHABET_COOLDOWN
    ) {

        sentence += letter;

        lastAlphabetAddTime = now;

        alphabetStableCount = 0;

        updateSentence();
    }
}


// ================= WORD PROCESSING =================

function processWord(word, confidence) {

    // Ignore invalid predictions
    if (
        !word ||
        word === "Unknown" ||
        word === "No hand detected" ||
        word === "Waiting..." ||
        word === "Collecting movement..." ||
        confidence < 0.75
    ) {

        wordStableCount = 0;
        return;
    }


    // Convert to clean text
    word = String(word)
        .trim()
        .toUpperCase();


    // Check whether same word continues
    if (word === lastDetectedWord) {

        wordStableCount++;

    } else {

        lastDetectedWord = word;
        wordStableCount = 1;
    }


    const now = Date.now();


    // Add word only after stable detection
    if (
        wordStableCount >= WORD_CONFIRM_FRAMES &&

        word !== lastAddedWord &&

        now - lastWordAddTime >= WORD_COOLDOWN
    ) {

        // Add space before the next word
        if (
            sentence.length > 0 &&
            !sentence.endsWith(" ")
        ) {

            sentence += " ";
        }


        sentence += word;

        lastAddedWord = word;

        lastWordAddTime = now;

        wordStableCount = 0;

        updateSentence();
    }
}


// ================= ADD SPACE =================

function addSpace() {

    if (
        sentence.length &&
        !sentence.endsWith(" ")
    ) {

        sentence += " ";

        updateSentence();
    }

    resetAlphabetLock();
    resetWordLock();
}


// ================= DELETE =================

function deleteLetter() {

    sentence = sentence.slice(0, -1);

    updateSentence();

    resetAlphabetLock();
    resetWordLock();
}


// ================= CLEAR =================

function clearSentence() {

    sentence = "";

    updateSentence();

    resetAlphabetLock();
    resetWordLock();
}


// ================= RESET ALPHABET =================

function resetAlphabetLock() {

    lastAddedAlphabet = "";
    alphabetStableCount = 0;
    lastAlphabetAddTime = Date.now();
}


// ================= RESET WORD =================

function resetWordLock() {

    lastDetectedWord = "";
    wordStableCount = 0;
    lastAddedWord = "";
    lastWordAddTime = Date.now();
}


// ================= UPDATE SENTENCE =================

function updateSentence() {

    const box =
        document.getElementById("sentence");

    if (box) {

        box.innerText =
            sentence ||
            "Your sentence will appear here...";
    }
}


// ================= SPEAK SENTENCE =================

function speakSentence() {

    if (!sentence.trim()) {

        alert("Build a sentence first.");
        return;
    }


    window.speechSynthesis.cancel();


    const speech =
        new SpeechSynthesisUtterance(
            sentence.trim()
        );

    speech.lang = "en-US";
    speech.rate = 0.85;
    speech.pitch = 1;
    speech.volume = 1;


    window.speechSynthesis.speak(
        speech
    );
}


// ================= SPEAK WORD =================

function speakDetectedWord() {

    const text = detectedWord;


    if (
        text === "Unknown" ||
        text === "No hand detected" ||
        text === "Collecting movement..." ||
        text === "Waiting..."
    ) {

        alert(
            "Please perform a sign and wait for the word to be detected."
        );

        return;
    }


    window.speechSynthesis.cancel();


    const speech =
        new SpeechSynthesisUtterance(text);

    speech.lang = "en-US";
    speech.rate = 0.85;


    window.speechSynthesis.speak(
        speech
    );
}


// ================= STOP CAMERA =================

function stopCamera() {

    if (predictionInterval) {

        clearInterval(
            predictionInterval
        );

        predictionInterval = null;
    }


    if (stream) {

        stream.getTracks().forEach(
            track => track.stop()
        );

        stream = null;

        video.srcObject = null;
    }

    sequence = [];
    predictionState = null;
    noHandCount = 0;
    frameCounter = 0;
    updateFrameProgress();

    window.speechSynthesis.cancel();


    document.getElementById(
        "cameraMessage"
    ).innerText = "Camera stopped";


    const handStatus =
        document.getElementById(
            "handStatus"
        );

    if (handStatus) {

        handStatus.innerText =
            "🔴 No hand detected";
    }
}


// ================= CLEAR RESULT =================

function clearResult() {

    detectedWord = "Unknown";
    detectedAlphabet = "Unknown";


    window.speechSynthesis.cancel();


    document.getElementById(
        "word"
    ).innerText = "Waiting...";


    document.getElementById(
        "confidence"
    ).innerText = "0%";


    document.getElementById(
        "confidenceBar"
    ).style.width = "0%";


    const handStatus =
        document.getElementById(
            "handStatus"
        );

    if (handStatus) {

        handStatus.innerText =
            "🔴 No hand detected";
    }


    resetAlphabetLock();
    resetWordLock();
}


// ================= MODE CHANGE =================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        document
            .querySelectorAll(
                'input[name="detectMode"]'
            )
            .forEach(radio => {

                radio.addEventListener(
                    "change",
                    () => {

                        resetAlphabetLock();
                        resetWordLock();

                        const mode =
                            getMode();


                        document.getElementById(
                            "detectionLabel"
                        ).innerText =

                            mode === "alphabet"
                                ? "Detected Alphabet"

                            : mode === "word"
                                ? "Detected Word"

                            : "Detected Result";
                    }
                );
            });
    }
);