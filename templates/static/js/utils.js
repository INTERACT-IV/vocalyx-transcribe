// utils.js
// Utilitaires généraux

/**
 * Formate une date ISO en format lisible français
 */
function formatHumanDate(isoString) {
    if (!isoString) return '-';
    const d = new Date(isoString);
    return d.toLocaleString('fr-FR', {
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit', 
        minute: '2-digit', 
        second: '2-digit'
    });
}

/**
 * Affiche une notification toast
 * @param {string} message - Message à afficher
 * @param {string} type - Type de toast (success, error, warning, info)
 */
function showToast(message, type="success") {
    toastr.options = {
        positionClass: "toast-top-right",
        timeOut: 3000,
        progressBar: true,
        closeButton: true,
        newestOnTop: true
    };
    toastr[type](message);
}

/**
 * Met à jour l'heure actuelle dans le header
 */
function updateCurrentTime() {
    const now = new Date();
    const timeElement = document.getElementById("current-time");
    if (timeElement) {
        timeElement.textContent = now.toLocaleString('fr-FR', {
            year: 'numeric', 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit'
        });
    }
}

/**
 * Échappe les caractères HTML pour éviter les injections XSS
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}