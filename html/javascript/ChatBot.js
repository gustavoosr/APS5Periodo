const textarea = document.getElementById("message");
const messagaDiv = document.getElementById("chat");

textarea.addEventListener("input", function () {
    this.style.height = "auto";
    this.style.height = Math.min(this.scrollHeight, 160) + "px";
});

async function sendMessage() {
    const text = textarea.value.trim();
    if (text === "") return;

    const msgUser = document.createElement("div");
    msgUser.classList.add("message", "user")
    msgUser.textContent = text;
    messagaDiv.appendChild(msgUser);

    textarea.value = "";
    textarea.style.height = "auto";
    messagaDiv.scrollTop = messagaDiv.scrollHeight;
    try {
        const response = await fetch("http://localhost:3000/mensagem", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();

        // Adiciona a resposta da IA
        const msgResponse = document.createElement("div");
        msgResponse.classList.add("message", "response");
        msgResponse.textContent = data.response;
        messagaDiv.appendChild(msgResponse);
        messagaDiv.scrollTop = messagaDiv.scrollHeight;

    } catch (error) {
        const errorMsg = document.createElement("div");
        errorMsg.classList.add("message", "response");
        errorMsg.textContent = "Erro ao se comunicar com o servidor.";
        messagaDiv.appendChild(errorMsg);
        console.error("Erro:", error);
    }
}
