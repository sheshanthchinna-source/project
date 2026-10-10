let stream = null;
let predictionInterval = null;

let detectedWord = "Unknown";
let detectedAlphabet = "Unknown";

let busy = false;

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
const ctx = canvas.getContext("2d");


function getMode() {
    return document.querySelector(
        'input[name="detectMode"]:checked'
    )?.value || "both";
}


// ================= START CAMERA =================
async function startCamera() {
    try {

        stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        });

        video.srcObject = stream;

        document.getElementById("cameraMessage").innerText =
            "Camera started - Show your hand";

        video.onloadedmetadata = () => {

            canvas.width = 320;
            canvas.height = 240;

            if (predictionInterval) {
                clearInterval(predictionInterval);
            }

            predictionInterval = setInterval(sendFrame, 100);
        };

    } catch (error) {

        console.error("Camera error:", error);

        alert(
            "Unable to access camera. Please allow camera permission."
        );
    }
}


// ================= SEND FRAME =================
async function sendFrame() {

    if (
        !stream ||
        busy ||
        video.readyState !== 4
    ) {
        return;
    }

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    const imageData = canvas.toDataURL(
        "image/jpeg",
        0.7
    );

    busy = true;

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image: imageData
            })
        });

        const result = await response.json();

        if (result.error || result.skipped) {
            return;
        }


        // ================= GET PREDICTIONS =================

        detectedWord =
            result.word || "Unknown";

        detectedAlphabet =
            result.alphabet || "Unknown";


        const mode = getMode();


        // ================= DISPLAY RESULT =================

        let displayPrediction = "Waiting...";

        if (mode === "alphabet") {

            displayPrediction = detectedAlphabet;

        } else if (mode === "word") {

            displayPrediction = detectedWord;

        } else {

            // BOTH MODE
            if (
                detectedWord !== "Unknown" &&
                detectedWord !== ""
            ) {
                displayPrediction = detectedWord;
            } else {
                displayPrediction = detectedAlphabet;
            }
        }


        document.getElementById("word").innerText =
            displayPrediction;


        // ================= CONFIDENCE =================

        let confidence = 0;

        if (mode === "alphabet") {

            confidence =
                (result.alphabet_confidence || 0) * 100;

        } else {

            confidence =
                (result.confidence || 0) * 100;
        }


        document.getElementById("confidence").innerText =
            confidence.toFixed(1) + "%";


        document.getElementById("confidenceBar").style.width =
            Math.min(confidence, 100) + "%";


        // ================= HAND STATUS =================

        const handStatus =
            document.getElementById("handStatus");

        if (handStatus) {

            handStatus.innerText =
                result.hand_detected
                    ? "🟢 Hand detected"
                    : "🔴 No hand detected";
        }


        // ================= SENTENCE BUILDER =================

        if (
            mode === "alphabet" &&
            result.hand_detected
        ) {

            processAlphabet(
                detectedAlphabet,
                result.alphabet_confidence || 0
            );

        } else if (
            mode === "word" &&
            result.hand_detected
        ) {

            processWord(
                detectedWord,
                result.confidence || 0
            );

        } else if (
            mode === "both" &&
            result.hand_detected
        ) {

            /*
             * In BOTH mode:
             * First try word detection.
             * If no valid word is available,
             * use alphabet detection.
             */

            if (
                detectedWord &&
                detectedWord !== "Unknown" &&
                detectedWord !== "No hand detected"
            ) {

                processWord(
                    detectedWord,
                    result.confidence || 0
                );

            } else {

                processAlphabet(
                    detectedAlphabet,
                    result.alphabet_confidence || 0
                );
            }
        }


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );

    } finally {

        busy = false;
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