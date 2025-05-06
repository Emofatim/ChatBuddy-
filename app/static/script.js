let mediaRecorder;
let audioChunks = [];

const chatBox = document.getElementById("chatBox");
const recordButton = document.getElementById("recordBtn");

recordButton.addEventListener("click", async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const formData = new FormData();
            formData.append("audio", audioBlob, "input.wav");

            chatBox.innerHTML += `<div class="user">🗣️ You said something...</div>`;

            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            const emotion = data.emotion;
            const reply = data.response;

            chatBox.innerHTML += `<div class="bot">🤖 (${emotion}): ${reply}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;
        };

        mediaRecorder.start();
        chatBox.innerHTML += `<div class="bot">🎤 Listening... Press ENTER to stop.</div>`;
        chatBox.scrollTop = chatBox.scrollHeight;

        document.onkeydown = (event) => {
            if (event.key === "Enter" && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
        };
    } catch (err) {
        console.error("Microphone error:", err);
        chatBox.innerHTML += `<div class="bot">❌ Please allow microphone access.</div>`;
    }
});
