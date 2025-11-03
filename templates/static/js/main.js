// main.js
// Point d'entrée principal de l'application

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    // Démarrer la mise à jour de l'heure
    setInterval(updateCurrentTime, 1000);
    updateCurrentTime();
    
    // Charger les cartes initiales
    refreshCards(1, currentLimit);
    
    // Démarrer le polling
    startPolling();
});