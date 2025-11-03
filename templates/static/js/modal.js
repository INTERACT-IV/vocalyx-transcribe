// modal.js
// Gestion de la modal

const modal = document.getElementById("modal");
const modalBody = document.getElementById("modal-body");
const spanClose = document.querySelector(".close");

/**
 * Ouvre la modal
 */
function openModal() {
    modal.style.display = "block";
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", handleKeyDown);
}

/**
 * Ferme la modal
 */
function closeModal() {
    modal.style.display = "none";
    document.body.style.overflow = "";
    document.removeEventListener("keydown", handleKeyDown);
}

/**
 * Gestion des touches clavier
 */
function handleKeyDown(e) {
    if (e.key === "Escape") closeModal();
}

// Événements de fermeture
spanClose.onclick = closeModal;
window.onclick = (event) => {
    if (event.target == modal) closeModal();
};

/**
 * Rend le contenu de la modal avec les détails d'une transcription
 * @param {Object} data - Données de la transcription
 */
function renderTranscriptionModal(data) {
    const segmentsHtml = (data.segments || []).map(seg => `
        <div class="segment" data-start="${seg.start}" tabindex="0">
            <strong>[${seg.start} - ${seg.end}]</strong> ${escapeHtml(seg.text || '')}
        </div>
    `).join('');
    
    modalBody.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
            <h2>Détails de la transcription</h2>
            <button class="btn btn-primary" onclick="closeModal()">← Retour</button>
        </div>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
            <div style="flex:1;min-width:200px;background:#f9f9f9;padding:1rem;border-radius:12px;box-shadow:0 4px 12px rgba(0,0,0,0.08);">
                <h4>Informations</h4>
                <div style="display:flex;flex-wrap:wrap;gap:1rem;">
                    <div style="flex:1;">
                        <p><strong>ID:</strong> ${escapeHtml(data.id || '-')}</p>
                        <p><strong>Statut:</strong> ${escapeHtml(data.status || '-')}</p>
                        <p><strong>Langue:</strong> ${escapeHtml(data.language || '-')}</p>
                        <p><strong>VAD:</strong> ${data.vad_enabled ? '✅ Activé' : '❌ Désactivé'}</p>
                    </div>
                    <div style="flex:1;">
                        <p><strong>Temps de traitement:</strong> ${data.processing_time ?? '-'}s</p>
                        <p><strong>Durée audio:</strong> ${data.duration ?? '-'}s</p>
                        <p><strong>Segments:</strong> ${data.segments_count || 0}</p>
                        <p><strong>Créé:</strong> ${formatHumanDate(data.created_at)}</p>
                        <p><strong>Terminé:</strong> ${formatHumanDate(data.finished_at)}</p>
                    </div>
                </div>
            </div>
        </div>
        <div style="display:flex;gap:1rem;flex-wrap:wrap;">
            <div class="segments-panel" style="width:280px;max-height:70vh;overflow-y:auto;background:white;padding:1rem;border-radius:12px;box-shadow:0 6px 12px rgba(0,0,0,0.1);">
                <h3>Segments (${data.segments_count || 0})</h3>
                ${segmentsHtml || '<p>Aucun segment disponible.</p>'}
            </div>
            <div class="text-panel" style="flex:1;max-height:70vh;overflow-y:auto;background:white;padding:1rem;border-radius:12px;box-shadow:0 6px 12px rgba(0,0,0,0.1);">
                <h3>Texte complet</h3>
                <p style="white-space:pre-wrap;">${escapeHtml(data.text || 'Aucun texte disponible.')}</p>
                ${data.error_message ? `<div style="margin-top:1rem;padding:1rem;background:#fee;border-radius:8px;color:#c00;">
                    <strong>❌ Erreur:</strong> ${escapeHtml(data.error_message)}
                </div>` : ''}
                <div style="margin-top:1rem;display:flex;gap:0.5rem;">
                    <a href="/transcribe/${data.id}" target="_blank" class="btn btn-primary">Voir le JSON</a>
                    <button onclick="copyToClipboard('${escapeHtml(data.text || '')}')" class="btn btn-success">📋 Copier</button>
                </div>
            </div>
        </div>
    `;
    
    // Ajouter les événements de clic sur les segments
    document.querySelectorAll(".segment").forEach(seg => {
        seg.addEventListener("click", () => {
            document.querySelectorAll(".segment").forEach(s => s.style.background = "");
            seg.style.background = "#e9f7fe";
        });
    });
}

/**
 * Copie le texte dans le presse-papier
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast("Texte copié dans le presse-papier !", "success");
    }).catch(err => {
        showToast("Erreur lors de la copie", "error");
    });
}