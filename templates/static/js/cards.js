// cards.js
// Gestion des cartes de transcription

let currentPage = 1;
let currentLimit = 10;

function attachDetailEvents() {
    document.querySelectorAll(".btn-detail").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const card = e.target.closest(".card");
            const id = card.dataset.id;
            openModal();
            modalBody.innerHTML = `
                <div style="text-align:center;padding:2rem;">
                    <div class="spinner"></div>
                    <p>Chargement des détails...</p>
                </div>
            `;
            try {
                const resp = await fetch(`/api/transcribe/${id}`);
                if (!resp.ok) throw new Error(`Erreur: ${resp.status}`);
                const data = await resp.json();
                renderTranscriptionModal(data);
            } catch (err) {
                modalBody.innerHTML = `
                    <div style="text-align:center;padding:2rem;color:red;">
                        <p>❌ Erreur: ${err.message}</p>
                        <button onclick="closeModal()" class="btn btn-danger">Fermer</button>
                    </div>
                `;
            }
        });
    });
}

function attachDeleteEvents() {
    document.querySelectorAll(".btn-delete").forEach(btn => {
        btn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const card = e.target.closest(".card");
            const id = card.dataset.id;
            if (!confirm(`Supprimer la transcription ${id.substring(0, 8)}... ?`)) return;
            
            try {
                const resp = await fetch(`/api/transcribe/${id}`, { method: "DELETE" });
                if (!resp.ok) throw new Error(await resp.text());
                showToast(`Transcription supprimée !`, "success");
                
                // Animation de suppression
                card.style.transition = "opacity 0.3s, transform 0.3s";
                card.style.opacity = "0";
                card.style.transform = "scale(0.8)";
                
                setTimeout(() => {
                    refreshCards(currentPage, currentLimit);
                }, 300);
            } catch (err) {
                showToast(`Erreur: ${err.message}`, "error");
            }
        });
    });
}

async function refreshCards(page=1, limit=10) {
    const status = document.getElementById("status-filter").value;
    const search = document.getElementById("search-input").value;
    
    currentPage = page;
    currentLimit = limit;
    
    try {
        let url = `/api/transcribe/recent?limit=${limit}&page=${page}`;
        if (status) url += `&status=${status}`;
        if (search) url += `&search=${encodeURIComponent(search)}`;
        
        const resp = await fetch(url);
        if (!resp.ok) throw new Error(`HTTP error! status: ${resp.status}`);
        const entries = await resp.json();
        
        // Récupérer le total pour la pagination
        let countUrl = `/api/transcribe/count?`;
        if (status) countUrl += `status=${status}&`;
        if (search) countUrl += `search=${encodeURIComponent(search)}`;
        
        const countResp = await fetch(countUrl);
        const countData = await countResp.json();
        const totalPages = Math.ceil(countData.total / limit);
        
        const container = document.getElementById("cards-container");
        container.innerHTML = "";
        
        if (entries.length === 0) {
            container.innerHTML = `<div style="text-align:center;width:100%;padding:2rem;">Aucune transcription trouvée.</div>`;
            updatePagination(page, 0);
            return;
        }
        
        const fragment = document.createDocumentFragment();
        entries.forEach((entry, i) => {
            const card = document.createElement("div");
            card.className = `card status-${entry.status || 'unknown'}`;
            card.dataset.id = entry.id;
            card.innerHTML = `
                <div class="status-badge"></div>
                <h3>ID: ${entry.id.substring(0, 8)}...</h3>
                <p><strong>Statut:</strong> <span class="status-text">${entry.status || '-'}</span></p>
                <p><strong>Langue:</strong> ${entry.language || 'inconnu'}</p>
                <p><strong>Durée:</strong> ${entry.duration ?? '-'}s</p>
                <p><strong>Créé:</strong> ${formatHumanDate(entry.created_at)}</p>
                <p><strong>Aperçu:</strong> ${entry.text ? entry.text.substring(0, 100) + (entry.text.length > 100 ? '...' : '') : '-'}</p>
                <div class="card-buttons">
                    <button class="btn-detail btn btn-primary">Détails</button>
                    <button class="btn-delete btn btn-danger">Supprimer</button>
                </div>
            `;
            fragment.appendChild(card);
            setTimeout(() => card.classList.add('visible'), i * 50);
        });
        
        container.appendChild(fragment);
        attachDetailEvents();
        attachDeleteEvents();
        updatePagination(page, totalPages);
        
    } catch (err) {
        console.error("Erreur:", err);
        document.getElementById("cards-container").innerHTML =
            `<div style="color:red;text-align:center;padding:2rem;">Erreur de chargement. Veuillez réessayer.</div>`;
    }
}

function updatePagination(currentPage, totalPages) {
    const pagination = document.getElementById("pagination");
    pagination.innerHTML = "";
    
    if (totalPages <= 1) return;
    
    for (let i = 1; i <= totalPages; i++) {
        const btn = document.createElement("button");
        btn.textContent = i;
        btn.dataset.page = i;
        if (i === currentPage) btn.classList.add("active");
        btn.addEventListener("click", () => {
            refreshCards(i, currentLimit);
        });
        pagination.appendChild(btn);
    }
}