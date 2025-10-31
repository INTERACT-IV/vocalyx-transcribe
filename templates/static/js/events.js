// events.js
// Gestion des événements utilisateur

document.getElementById("upload-btn").addEventListener("click", () => {
    document.getElementById("upload-input").click();
});

document.getElementById("upload-input").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append("file", file);
    
    const loadingOverlay = document.getElementById("loading-overlay");
    loadingOverlay.style.display = "flex";
    
    try {
        const resp = await fetch("/api/transcribe", {
            method: "POST",
            body: formData
        });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        showToast(`✅ Upload réussi ! ID: ${data.transcription_id.substring(0, 8)}...`, "success");
        
        // Rafraîchir immédiatement
        await refreshCards(1, currentLimit);
        
    } catch (err) {
        showToast(`❌ Erreur: ${err.message}`, "error");
    } finally {
        loadingOverlay.style.display = "none";
        event.target.value = "";
    }
});

document.getElementById("export-btn").addEventListener("click", async () => {
    try {
        const resp = await fetch("/api/transcribe/export");
        if (!resp.ok) throw new Error(`Erreur: ${resp.status}`);
        const data = await resp.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `transcriptions_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        showToast("Export terminé !", "success");
    } catch (err) {
        showToast(`Erreur: ${err.message}`, "error");
    }
});

document.getElementById("status-filter").addEventListener("change", () => {
    refreshCards(1, currentLimit);
});

document.getElementById("search-input").addEventListener("input", () => {
    refreshCards(1, currentLimit);
});