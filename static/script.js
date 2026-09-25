let stream = null;
let predictionInterval = null;

let detectedWord = "Unknown";


const video =
    document.getElementById("video");

const canvas =
    document.getElementById("canvas");

const ctx =
    canvas.getContext("2d");


async function startCamera() {

    try {

        stream =
            await navigator.mediaDevices
                .getUserMedia({

                    video: true,
                    audio: false

                });


        video.srcObject =
            stream;


        document.getElementById(
            "cameraMessage"
        ).innerText =
            "Camera started - Show your hand";


        // Wait for video

        video.onloadedmetadata = () => {

            canvas.width =
                video.videoWidth;

            canvas.height =
                video.videoHeight;


            // Send frames to Flask

            predictionInterval =
                setInterval(
                    sendFrame,
                    150
                );

        };

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );


        alert(
            "Unable to access camera. Please allow camera permission."
        );

    }

}


async function sendFrame() {

    if (
        !stream ||
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


    const imageData =
        canvas.toDataURL(
            "image/jpeg",
            0.7
        );


    try {

        const response =
            await fetch(
                "/predict",
                {

                    method: "POST",

                    headers: {

                        "Content-Type":
                            "application/json"

                    },

                    body: JSON.stringify({

                        image: imageData

                    })

                }
            );


        const result =
            await response.json();


        detectedWord =
            result.word;


        document.getElementById(
            "word"
        ).innerText =
            result.word;


        const confidence =
            result.confidence * 100;


        document.getElementById(
            "confidence"
        ).innerText =
            confidence.toFixed(1) + "%";


        document.getElementById(
            "confidenceBar"
        ).style.width =
            Math.min(
                confidence,
                100
            ) + "%";


        const handStatus =
            document.getElementById(
                "handStatus"
            );


        if (handStatus) {

            if (result.hand_detected) {

                handStatus.innerText =
                    "🟢 Hand detected";

            }

            else {

                handStatus.innerText =
                    "🔴 No hand detected";

            }

        }

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

    }

}


function speakDetectedWord() {

    // Don't speak invalid results

    if (
        detectedWord === "Unknown" ||
        detectedWord === "No hand detected" ||
        detectedWord === "Collecting movement..." ||
        detectedWord === "Waiting..."
    ) {

        alert(
            "Please perform a sign and wait for the word to be detected."
        );

        return;

    }


    // Stop previous speech

    window.speechSynthesis.cancel();


    // Create speech

    const speech =
        new SpeechSynthesisUtterance(
            detectedWord
        );


    speech.lang =
        "en-US";


    speech.rate =
        0.85;


    speech.pitch =
        1;


    speech.volume =
        1;


    // Speak

    window.speechSynthesis.speak(
        speech
    );

}


function stopCamera() {

    // Stop prediction interval

    if (predictionInterval) {

        clearInterval(
            predictionInterval
        );

        predictionInterval =
            null;

    }


    // Stop camera tracks

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );


        stream =
            null;


        video.srcObject =
            null;

    }


    // Stop speech

    window.speechSynthesis.cancel();


    // Reset camera message

    document.getElementById(
        "cameraMessage"
    ).innerText =
        "Camera stopped";


    // Reset hand status

    const handStatus =
        document.getElementById(
            "handStatus"
        );


    if (handStatus) {

        handStatus.innerText =
            "🔴 No hand detected";

    }

}



function clearResult() {


    detectedWord =
        "Unknown";


    window.speechSynthesis.cancel();

    document.getElementById(
        "word"
    ).innerText =
        "Waiting...";

    document.getElementById(
        "confidence"
    ).innerText =
        "0%";

    document.getElementById(
        "confidenceBar"
    ).style.width =
        "0%";

    const handStatus =
        document.getElementById(
            "handStatus"
        );


    if (handStatus) {

        handStatus.innerText =
            "🔴 No hand detected";

    }

}