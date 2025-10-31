// polling.js
// Gestion du polling intelligent

let pollingInterval = null;

function startPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    
    pollingInterval = setInterval(async () => {
        // Ne pas rafraîchir si la modal est ouverte
        if (modal.style.display === "block") return;
        
        // Rafraîchir sans animation pour le polling
        const status = document.getElementById("status-filter").value;
        const search = document.getElementById("search-input").value;
        
        try {
            let url = `/api/transcribe/recent?limit=${currentLimit}&page=${currentPage}`;
            if (status) url += `&status=${status}`;
            if (search) url += `&search=${encodeURIComponent(search)}`;
            
            const resp = await fetch(url);
            if (!resp.ok) return;
            const entries = await resp.json();
            
            const container = document.getElementById("cards-container");
            const existingIds = new Set(
                Array.from(container.querySelectorAll('.card')).map(c => c.dataset.id)
            );
            
            const newIds = new Set(entries.map(e => e.id));
            
            // Vérifier s'il y a des changements
            const hasChanges = 
                existingIds.size !== newIds.size || 
                ![...existingIds].every(id => newIds.has(id));
            
            if (hasChanges) {
                console.log('🔄 Changements détectés, rafraîchissement...');
                await refreshCards(currentPage, currentLimit);
            } else {
                // Mettre à jour uniquement les statuts si nécessaire
                entries.forEach(entry => {
                    const card = container.querySelector(`[data-id="${entry.id}"]`);
                    if (card) {
                        const statusText = card.querySelector('.status-text');
                        if (statusText && statusText.textContent !== entry.status) {
                            // Statut changé, rafraîchir complètement
                            refreshCards(currentPage, currentLimit);
                        }
                    }
                });
            }
        } catch (err) {
            console.error('Erreur polling:', err);
        }
    }, 5000); // Polling toutes les 5 secondes
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
}

// Arrêter le polling quand la page est cachée
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopPolling();
    } else {
        startPolling();
    }
});