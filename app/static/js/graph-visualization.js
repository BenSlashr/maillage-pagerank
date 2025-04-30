/**
 * Visualisation du maillage interne et du PageRank
 * 
 * Ce script gère la visualisation du graphe de maillage interne
 * et l'affichage des scores PageRank.
 */

// Configuration globale
const config = {
    nodeSize: {
        min: 5,
        max: 30,
        factor: 100
    },
    colors: {
        nodes: {
            default: '#6c757d',
            category: '#007bff',
            product: '#28a745',
            blog: '#fd7e14'
        },
        edges: {
            existing: '#6c757d',
            suggested: '#28a745'
        }
    }
};

// Stockage des données
let graphData = null;
let pagerankData = null;
let contentLinksOnly = false; // Option pour filtrer les liens par position
let useWeightedPagerank = false; // Option pour utiliser le PageRank pondéré
let alpha = 0.5; // Coefficient pour la pondération sémantique
let beta = 0.5; // Coefficient pour la pondération par position
let currentNetwork = null;
let optimizedNetwork = null;

/**
 * Initialise la visualisation
 * @param {string} jobId - ID de la tâche d'analyse
 */
async function initVisualization(jobId) {
    try {
        // Afficher le message de chargement
        document.getElementById('loading-message').style.display = 'block';
        
        // Charger les données du graphe
        await loadGraphData(jobId);
        
        // Charger les données de PageRank
        await loadPagerankData(jobId);
        
        // Vérifier si nous avons des données à afficher ou s'il y a une erreur
        if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
            let errorMessage = 'Aucune donnée de graphe disponible pour ce job.';
            
            // Afficher le message d'erreur spécifique s'il existe
            if (graphData && graphData.error) {
                errorMessage = graphData.error;
            } else {
                errorMessage += '<br>Vérifiez que vous avez bien fourni un fichier de liens avec des colonnes "Source" et "Destination".';
            }
            
            document.getElementById('loading-message').innerHTML = `
                <div class="alert alert-warning">
                    <h4 class="alert-heading">Visualisation non disponible</h4>
                    <p>${errorMessage}</p>
                    <hr>
                    <p class="mb-0">Pour visualiser le maillage interne, assurez-vous d'avoir fourni un fichier de liens valide lors de l'analyse.</p>
                </div>
            `;
            return;
        }
        
        // Créer les visualisations
        createCurrentVisualization();
        createOptimizedVisualization();
        
        // Afficher les statistiques
        displayStatistics();
        
        // Initialiser les onglets (après avoir créé les visualisations)
        initTabs();
        
        // Masquer le message de chargement
        document.getElementById('loading-message').style.display = 'none';
        
        // Afficher le contenu principal
        document.getElementById('visualization-container').style.display = 'block';
    } catch (error) {
        console.error('Erreur lors de l\'initialisation de la visualisation:', error);
        document.getElementById('loading-message').textContent = `Erreur: ${error.message}`;
    }
}

/**
 * Charge les données du graphe depuis l'API
 * @param {string} jobId - ID de la tâche d'analyse
 */
async function loadGraphData(jobId) {
    try {
        const response = await fetch(`${basePath}/api/graph/${jobId}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Erreur lors du chargement des données du graphe');
        }
        graphData = await response.json();
        console.log('Données du graphe chargées:', graphData);
    } catch (error) {
        console.error('Erreur lors du chargement des données du graphe:', error);
        throw error;
    }
}

/**
 * Charge les données de PageRank depuis l'API
 * @param {string} jobId - ID de la tâche d'analyse
 */
async function loadPagerankData(jobId) {
    try {
        // Utiliser les paramètres de configuration actuels
        const { useWeightedPagerank, contentLinksOnly, alpha, beta } = pagerankConfig;
        
        // Construire l'URL avec les paramètres
        let url;
        if (useWeightedPagerank) {
            url = `${basePath}/api/weighted-pagerank/${jobId}?content_links_only=${contentLinksOnly}&alpha=${alpha}&beta=${beta}`;
        } else {
            url = `${basePath}/api/pagerank/${jobId}?content_links_only=${contentLinksOnly}`;
        }
        
        console.log(`Chargement du PageRank avec les paramètres: useWeightedPagerank=${useWeightedPagerank}, contentLinksOnly=${contentLinksOnly}, alpha=${alpha}, beta=${beta}`);
        
        // Ajouter un timestamp pour éviter la mise en cache
        url += `&_=${new Date().getTime()}`;
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        pagerankData = await response.json();
        
        console.log('Données PageRank reçues:', pagerankData);
        
        // Afficher un message de notification pour le type de PageRank utilisé
        showNotification(useWeightedPagerank ? 
            `PageRank pondéré appliqué (α=${alpha}, β=${beta})${contentLinksOnly ? ', liens de contenu uniquement' : ''}` : 
            `PageRank standard${contentLinksOnly ? ' (liens de contenu uniquement)' : ''}`);
    } catch (error) {
        console.error('Erreur lors du chargement des données PageRank:', error);
        pagerankData = null;
        showNotification('Erreur lors du chargement des données PageRank', 'danger');
    }
}

/**
 * Initialise les onglets pour basculer entre les différentes vues
 */
function initTabs() {
    // Événement de changement d'onglet
    document.querySelectorAll('a[data-bs-toggle="tab"]').forEach(tab => {
        tab.addEventListener('shown.bs.tab', function (event) {
            // Activer l'onglet cliqué
            this.classList.add('active');
            
            // Mettre à jour la visualisation si nécessaire
            const target = this.getAttribute('href');
            if (target === '#current-network' && currentNetwork) {
                setTimeout(() => currentNetwork.fit(), 50);
            } else if (target === '#optimized-network' && optimizedNetwork) {
                setTimeout(() => optimizedNetwork.fit(), 50);
            } else if (target === '#pagerank-tab') {
                displayPagerankTable();
            }
        });
    });
}

/**
 * Génère les options de visualisation optimisées en fonction de la taille du graphe
 * @param {Array} nodes - Liste des nœuds
 * @param {Array} edges - Liste des liens
 * @returns {Object} - Options de visualisation optimisées
 */
function getNetworkOptions(nodes, edges) {
    // Optimiser les options pour les grands graphes
    const nodeCount = nodes.length;
    const edgeCount = edges.length;
    const isLargeGraph = nodeCount > 500;
    const isVeryLargeGraph = nodeCount > 1000;
    
    // Afficher un message d'information sur la taille du graphe
    console.log(`Création d'un graphe avec ${nodeCount} nœuds et ${edgeCount} liens`);
    
    return {
        nodes: {
            shape: 'dot',
            scaling: {
                min: config.nodeSize.min,
                max: config.nodeSize.max,
                label: { enabled: !isLargeGraph } // Désactiver les labels pour les grands graphes
            },
            font: {
                size: 12,
                face: 'Arial'
            }
        },
        edges: {
            width: 1,
            smooth: {
                type: isLargeGraph ? 'continuous' : 'dynamic',
                roundness: 0.5
            },
            arrows: {
                to: { enabled: true, scaleFactor: 0.5 }
            }
        },
        physics: {
            stabilization: {
                enabled: true,
                iterations: isLargeGraph ? 100 : 1000, // Moins d'itérations pour les grands graphes
                updateInterval: isLargeGraph ? 50 : 100
            },
            barnesHut: {
                gravitationalConstant: -80,
                springConstant: 0.001,
                springLength: 200
            }
        },
        layout: {
            improvedLayout: false // Désactiver l'algorithme de mise en page amélioré pour les grands graphes
        },
        interaction: {
            navigationButtons: true,
            keyboard: true,
            tooltipDelay: 200,
            hover: !isLargeGraph, // Désactiver le survol pour les grands graphes
            hideEdgesOnDrag: isLargeGraph, // Cacher les liens pendant le déplacement pour améliorer les performances
            hideNodesOnDrag: isVeryLargeGraph // Cacher les nœuds pendant le déplacement pour les très grands graphes
        }
    };
}

/**
 * Crée la visualisation du graphe actuel
 */
function createCurrentVisualization() {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
        console.warn('Aucune donnée de graphe disponible pour la visualisation actuelle');
        return;
    }
    
    // Vérifier si les données du graphe sont valides
    if (graphData.nodes.length === 0) {
        console.warn('Le graphe ne contient aucun nœud');
        return;
    }
    
    // Préparer les données pour vis.js
    const nodes = prepareNodes();
    const edges = prepareEdges('current');
    
    // Créer le réseau
    const container = document.getElementById('network-graph');
    if (!container) {
        console.error('Conteneur de graphe actuel introuvable');
        return;
    }
    
    const data = { nodes, edges };
    const options = getNetworkOptions(nodes, edges);
    
    // Créer le réseau
    currentNetwork = new vis.Network(container, data, options);
    
    // Événement de sélection d'un nœud
    currentNetwork.on('selectNode', function(params) {
        displayNodeDetails(params.nodes[0]);
    });
    
    // Événement de désélection
    currentNetwork.on('deselectNode', function() {
        hideNodeDetails();
    });
}

/**
 * Crée la visualisation du graphe optimisé
 */
function createOptimizedVisualization() {
    if (!graphData || !graphData.nodes || graphData.nodes.length === 0) {
        console.warn('Aucune donnée de graphe disponible pour la visualisation optimisée');
        return;
    }
    
    // Vérifier si les données du graphe sont valides
    if (graphData.nodes.length === 0) {
        console.warn('Le graphe ne contient aucun nœud');
        return;
    }
    
    // Préparer les données pour vis.js
    const nodes = prepareNodes();
    const edges = prepareEdges('optimized');
    
    // Créer le réseau
    const container = document.getElementById('optimized-graph');
    if (!container) {
        console.error('Conteneur de graphe optimisé introuvable');
        return;
    }
    
    const data = { nodes, edges };
    const options = getNetworkOptions(nodes, edges);
    
    // Créer le réseau
    optimizedNetwork = new vis.Network(container, data, options);
    
    // Événement de sélection d'un nœud
    optimizedNetwork.on('selectNode', function(params) {
        displayNodeDetails(params.nodes[0]);
    });
    
    // Événement de désélection
    optimizedNetwork.on('deselectNode', function() {
        hideNodeDetails();
    });
}

// Configuration pour le filtrage des nœuds
let nodeDisplayConfig = {
    maxNodes: 100,  // Nombre maximum de nœuds à afficher par défaut
    filterBy: 'pagerank'  // Critère de filtrage ('pagerank', 'links', 'all')
};

/**
 * Prépare les nœuds pour la visualisation avec filtrage
 * @param {number} maxNodes - Nombre maximum de nœuds à afficher (optionnel)
 * @returns {vis.DataSet} - Ensemble de données des nœuds filtrés
 */
function prepareNodes(maxNodes = nodeDisplayConfig.maxNodes) {
    if (!graphData || !graphData.nodes) return new vis.DataSet([]);
    
    // Préparer tous les nœuds avec leurs métriques
    let allNodes = graphData.nodes.map(node => {
        // Déterminer la taille du nœud en fonction du PageRank
        let value = config.nodeSize.min;
        let pagerank = 0;
        
        if (pagerankData && pagerankData.current && node.id in pagerankData.current) {
            pagerank = pagerankData.current[node.id].pagerank;
            value = pagerank * config.nodeSize.factor;
        }
        
        // Déterminer la couleur en fonction du type
        let color = config.colors.nodes.default;
        if (node.type in config.colors.nodes) {
            color = config.colors.nodes[node.type];
        }
        
        return {
            id: node.id,
            label: node.label,
            title: node.id,
            value: value,
            color: color,
            pagerank: pagerank,
            originalData: node
        };
    });
    
    // Filtrer les nœuds si nécessaire
    let filteredNodes = allNodes;
    if (maxNodes > 0 && maxNodes < allNodes.length) {
        // Trier par PageRank décroissant
        filteredNodes = allNodes.sort((a, b) => b.pagerank - a.pagerank).slice(0, maxNodes);
        console.log(`Affichage limité aux ${maxNodes} nœuds les plus importants (sur ${allNodes.length} total)`); 
    }
    
    return new vis.DataSet(filteredNodes);
}

/**
 * Met à jour le nombre maximum de nœuds à afficher
 * @param {number} newMaxNodes - Nouveau nombre maximum de nœuds
 */
function updateMaxNodes(newMaxNodes) {
    nodeDisplayConfig.maxNodes = newMaxNodes;
    // Mettre à jour les visualisations
    if (currentNetwork) {
        const nodes = prepareNodes();
        const edges = prepareEdges('current');
        currentNetwork.setData({ nodes, edges });
    }
    if (optimizedNetwork) {
        const nodes = prepareNodes();
        const edges = prepareEdges('optimized');
        optimizedNetwork.setData({ nodes, edges });
    }
}

/**
 * Prépare les arêtes pour la visualisation
 * @param {string} mode - Mode de visualisation ('current' ou 'optimized')
 * @param {Array} visibleNodeIds - Liste des IDs des nœuds visibles (optionnel)
 * @returns {vis.DataSet} - Ensemble de données des arêtes
 */
function prepareEdges(mode, visibleNodeIds = null) {
    if (!graphData || !graphData.edges) return new vis.DataSet([]);
    
    // Si aucune liste de nœuds visibles n'est fournie, utiliser les nœuds actuellement affichés
    if (!visibleNodeIds) {
        // Récupérer les nœuds actuellement affichés
        const currentNodes = prepareNodes();
        visibleNodeIds = currentNodes.getIds();
    }
    
    // Convertir en Set pour des recherches plus rapides
    const visibleNodeSet = new Set(visibleNodeIds);
    
    let edgeList = [];
    
    // Ajouter les liens existants (uniquement entre nœuds visibles)
    if (graphData.edges.current) {
        edgeList = graphData.edges.current
            .filter(edge => visibleNodeSet.has(edge.source) && visibleNodeSet.has(edge.target))
            .map(edge => ({
                from: edge.source,
                to: edge.target,
                color: config.colors.edges.existing,
                title: 'Lien existant',
                arrows: 'to',
                originalData: edge
            }));
    }
    
    // Ajouter les liens suggérés si on est en mode optimisé (uniquement entre nœuds visibles)
    if (mode === 'optimized' && graphData.edges.suggested) {
        const suggestedEdges = graphData.edges.suggested
            .filter(edge => visibleNodeSet.has(edge.source) && visibleNodeSet.has(edge.target))
            .map(edge => ({
                from: edge.source,
                to: edge.target,
                color: config.colors.edges.suggested,
                title: `Suggestion: ${edge.anchor || ''}`,
                arrows: 'to',
                dashes: true,
                originalData: edge
            }));
        
        edgeList = edgeList.concat(suggestedEdges);
    }
    
    console.log(`Affichage de ${edgeList.length} liens pour le mode ${mode}`);
    return new vis.DataSet(edgeList);
}

/**
 * Met à jour la visualisation en fonction du mode (ancienne fonction, maintenant dépréciée)
 * @param {string} mode - Mode de visualisation ('current' ou 'optimized')
 */
function updateVisualization(mode) {
    // Cette fonction est maintenant dépréciée car nous utilisons deux instances de graphe distinctes
    console.log(`Changement vers le mode ${mode}`);
}

/**
 * Affiche les détails d'un nœud sélectionné
 * @param {string} nodeId - ID du nœud sélectionné
 */
function displayNodeDetails(nodeId) {
    const detailsContainer = document.getElementById('node-details');
    const node = currentNetwork.body.data.nodes.get(nodeId);
    
    if (!node) return;
    
    let pagerankInfo = '';
    if (pagerankData && pagerankData.current && nodeId in pagerankData.current) {
        const currentRank = pagerankData.current[nodeId];
        pagerankInfo = `<p><strong>PageRank actuel:</strong> ${currentRank.pagerank.toFixed(6)} (Rang: ${currentRank.rank})</p>`;
        
        if (pagerankData.optimized && nodeId in pagerankData.optimized) {
            const optimizedRank = pagerankData.optimized[nodeId];
            const improvement = pagerankData.improvement[nodeId];
            
            pagerankInfo += `
                <p><strong>PageRank optimisé:</strong> ${optimizedRank.pagerank.toFixed(6)} (Rang: ${optimizedRank.rank})</p>
                <p><strong>Amélioration:</strong> ${improvement.percentage.toFixed(2)}% (${improvement.rank_change > 0 ? '+' : ''}${improvement.rank_change} positions)</p>
            `;
        }
    }
    
    detailsContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Détails de la page</h5>
            </div>
            <div class="card-body">
                <h6 class="card-title">${node.label}</h6>
                <p><strong>URL:</strong> <a href="${nodeId}" target="_blank">${nodeId}</a></p>
                <p><strong>Type:</strong> ${node.originalData.type}</p>
                ${pagerankInfo}
            </div>
        </div>
    `;
    
    detailsContainer.style.display = 'block';
}

/**
 * Masque les détails du nœud
 */
function hideNodeDetails() {
    document.getElementById('node-details').style.display = 'none';
}

/**
 * Affiche les statistiques du graphe
 */
function displayStatistics() {
    if (!graphData || !graphData.metrics) {
        console.warn('Aucune métrique disponible pour les statistiques');
        return;
    }
    
    const stats = graphData.metrics;
    const statsContainer = document.getElementById('graph-statistics');
    
    statsContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Statistiques du maillage</h5>
            </div>
            <div class="card-body">
                <p><strong>Nombre de pages:</strong> ${stats.node_count}</p>
                <p><strong>Liens existants:</strong> ${stats.current_edge_count}</p>
                <p><strong>Liens suggérés:</strong> ${stats.suggested_edge_count}</p>
                <p><strong>Amélioration potentielle:</strong> +${stats.improvement_percentage}%</p>
            </div>
        </div>
    `;
}

/**
 * Affiche le tableau des scores PageRank
 */
function displayPagerankTable(sortBy = null) {
    if (!pagerankData || !pagerankData.current) {
        console.warn('Aucune donnée PageRank disponible');
        return;
    }
    
    const tableContainer = document.getElementById('pagerank-table');
    const hasOptimized = pagerankData.optimized !== undefined;
    
    // Réinitialiser les données du tableau à chaque appel pour prendre en compte les nouvelles valeurs
    pagerankTableData = [];
    
    // Préparer les données pour le tableau avec les valeurs actuelles
    for (const url in pagerankData.current) {
        const currentData = pagerankData.current[url];
        const row = {
            url: url,
            current_pagerank: currentData.pagerank,
            current_rank: currentData.rank,
            optimized_pagerank: 0,
            optimized_rank: 0,
            improvement_percentage: 0,
            rank_change: 0
        };
        
        if (hasOptimized && url in pagerankData.optimized) {
            const optimizedData = pagerankData.optimized[url];
            
            row.optimized_pagerank = optimizedData.pagerank;
            row.optimized_rank = optimizedData.rank;
            
            // Calculer l'amélioration du PageRank directement dans le frontend
            const improvementPct = ((row.optimized_pagerank - row.current_pagerank) / row.current_pagerank) * 100;
            row.improvement_percentage = isNaN(improvementPct) ? 0 : improvementPct;
            
            // Calculer le changement de rang directement dans le frontend
            row.rank_change = row.current_rank - row.optimized_rank;
            
            // Utiliser les données du backend si disponibles
            if (pagerankData.improvement && url in pagerankData.improvement) {
                const improvement = pagerankData.improvement[url];
                if (improvement.percentage !== undefined) row.improvement_percentage = improvement.percentage;
                if (improvement.rank_change !== undefined) row.rank_change = improvement.rank_change;
            }
        }
        
        // Préparer l'URL d'affichage
        try {
            // Si c'est une URL complète, extraire le chemin
            if (url.startsWith('http')) {
                const urlObj = new URL(url);
                row.display_url = urlObj.pathname;
            } else {
                row.display_url = url;
            }
            
            // Si le chemin est trop long, le raccourcir intelligemment
            if (row.display_url.length > 50) {
                row.display_url = row.display_url.substring(0, 47) + '...';
            }
        } catch (e) {
            row.display_url = url;
            console.warn('Erreur lors du traitement de l\'URL:', e);
        }
        
        pagerankTableData.push(row);
    }
    
    // Mettre à jour la configuration de tri si nécessaire
    if (sortBy) {
        const [field, direction] = sortBy.split('_');
        pagerankSortConfig.field = field === 'current' ? 'current_pagerank' : 
                                  field === 'optimized' ? 'optimized_pagerank' : 
                                  'improvement_percentage';
        pagerankSortConfig.direction = direction;
    }
    
    // Appliquer le tri
    const sortedData = [...pagerankTableData].sort((a, b) => {
        const factor = pagerankSortConfig.direction === 'desc' ? -1 : 1;
        return factor * (a[pagerankSortConfig.field] - b[pagerankSortConfig.field]);
    });
    
    // Appliquer le filtre de recherche si présent
    const searchInput = document.getElementById('pagerank-search');
    let filteredData = sortedData;
    if (searchInput && searchInput.value) {
        const searchTerm = searchInput.value.toLowerCase();
        filteredData = sortedData.filter(row => 
            row.url.toLowerCase().includes(searchTerm) || 
            row.display_url.toLowerCase().includes(searchTerm)
        );
    }
    
    // Créer le tableau HTML
    let tableHTML = `
        <table class="table table-striped table-hover">
            <thead>
                <tr>
                    <th>URL</th>
                    <th>PageRank actuel</th>
                    <th>Rang actuel</th>
    `;
    
    if (hasOptimized) {
        tableHTML += `
                    <th>PageRank optimisé</th>
                    <th>Rang optimisé</th>
                    <th>Amélioration</th>
                    <th>Changement de rang</th>
        `;
    }
    
    tableHTML += `
                </tr>
            </thead>
            <tbody>
    `;
    
    // Afficher toutes les lignes filtrées
    if (filteredData.length === 0) {
        tableHTML += `
            <tr>
                <td colspan="${hasOptimized ? 7 : 3}" class="text-center">Aucun résultat trouvé</td>
            </tr>
        `;
    } else {
        for (const row of filteredData) {
            tableHTML += `
                <tr>
                    <td title="${row.url}">${row.display_url}</td>
                    <td>${row.current_pagerank.toFixed(6)}</td>
                    <td>${row.current_rank}</td>
            `;
            
            if (hasOptimized) {
                const improvementClass = row.improvement_percentage > 0 ? 'text-success' : (row.improvement_percentage < 0 ? 'text-danger' : '');
                const rankChangeClass = row.rank_change > 0 ? 'text-success' : (row.rank_change < 0 ? 'text-danger' : '');
                const rankChangeIcon = row.rank_change > 0 ? '↑' : (row.rank_change < 0 ? '↓' : '=');
                
                tableHTML += `
                    <td>${row.optimized_pagerank.toFixed(6)}</td>
                    <td>${row.optimized_rank}</td>
                    <td class="${improvementClass}">${row.improvement_percentage > 0 ? '+' : ''}${row.improvement_percentage.toFixed(2)}%</td>
                    <td class="${rankChangeClass}">${rankChangeIcon}${Math.abs(row.rank_change)}</td>
                `;
            }
            
            tableHTML += `
                </tr>
            `;
        }
    }
    
    tableHTML += `
            </tbody>
        </table>
    `;
    
    // Afficher des statistiques sur les résultats
    tableHTML += `
        <div class="mt-3 text-muted">
            Affichage de ${filteredData.length} pages sur ${pagerankTableData.length} au total.
        </div>
    `;
    
    tableContainer.innerHTML = tableHTML;
}

/**
 * Filtre le tableau PageRank en fonction de la recherche
 */
function filterPagerankTable() {
    displayPagerankTable();
}

/**
 * Met à jour le tri du tableau PageRank
 * @param {string} sortBy - Critère de tri
 */
function updatePagerankTable(sortBy) {
    displayPagerankTable(sortBy);
}

/**
 * Exporte le tableau PageRank au format spécifié
 * @param {string} format - Format d'export ('csv' ou 'excel')
 */
function exportPagerankTable(format) {
    if (pagerankTableData.length === 0) {
        alert('Aucune donnée à exporter');
        return;
    }
    
    // Préparer les données
    const hasOptimized = pagerankData.optimized !== undefined;
    let csvContent = 'URL,PageRank actuel,Rang actuel';
    
    if (hasOptimized) {
        csvContent += ',PageRank optimisé,Rang optimisé,Amélioration (%),Changement de rang';
    }
    csvContent += '\n';
    
    // Ajouter chaque ligne
    for (const row of pagerankTableData) {
        csvContent += `"${row.url}",${row.current_pagerank},${row.current_rank}`;
        
        if (hasOptimized) {
            csvContent += `,${row.optimized_pagerank},${row.optimized_rank},${row.improvement_percentage},${row.rank_change}`;
        }
        csvContent += '\n';
    }
    
    // Créer un lien de téléchargement
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `pagerank_export_${new Date().toISOString().slice(0, 10)}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Variables globales pour le tableau PageRank
let pagerankTableData = [];
let pagerankSortConfig = {
    field: 'current_pagerank',
    direction: 'desc'
};

/**
 * Active ou désactive le filtrage des liens par position (contenu uniquement)
 * @param {boolean} enabled - True pour activer le filtrage, false sinon
 */
async function toggleContentLinksOnly(enabled) {
    // Mettre à jour la configuration
    contentLinksOnly = enabled;
    
    // Afficher un message de notification
    showNotification(enabled ? 
        'Filtrage activé: liens de contenu uniquement' : 
        'Filtrage désactivé: tous les liens sont pris en compte');
    
    try {
        // Récupérer l'ID de la tâche depuis l'URL
        const jobId = getJobIdFromUrl();
        if (!jobId) return;
        
        // Recharger les données de PageRank avec la nouvelle configuration
        await loadPagerankData(jobId);
        
        // Mettre à jour les visualisations
        createCurrentVisualization();
        createOptimizedVisualization();
        
        // Mettre à jour les statistiques et le tableau PageRank
        displayStatistics();
        displayPagerankTable();
    } catch (error) {
        console.error('Erreur lors du changement de mode de filtrage:', error);
    }
}

/**
 * Active ou désactive le PageRank pondéré
 * @param {boolean} enabled - True pour activer le PageRank pondéré, false pour le PageRank standard
 */
function toggleWeightedPageRank(enabled) {
    // Mettre à jour la configuration
    useWeightedPagerank = enabled;
    
    // Afficher/masquer les contrôles de pondération
    const controls = document.getElementById('weighted-pagerank-controls');
    if (controls) {
        controls.classList.toggle('d-none', !enabled);
    }
    
    // Afficher un message de notification
    showNotification(enabled ? 
        'PageRank pondéré activé' : 
        'PageRank standard activé');
}

/**
 * Met à jour le coefficient alpha (pondération sémantique)
 * @param {number} value - Nouvelle valeur de alpha (entre 0 et 1)
 */
function updateAlpha(value) {
    alpha = parseFloat(value);
    document.getElementById('alpha-value').textContent = alpha.toFixed(1);
}

/**
 * Met à jour le coefficient beta (pondération par position)
 * @param {number} value - Nouvelle valeur de beta (entre 0 et 1)
 */
function updateBeta(value) {
    beta = parseFloat(value);
    document.getElementById('beta-value').textContent = beta.toFixed(1);
}

/**
 * Applique les pondérations au calcul du PageRank
 */
async function applyWeightedPageRank() {
    try {
        // Récupérer l'ID de la tâche depuis l'URL
        const jobId = getJobIdFromUrl();
        if (!jobId) return;
        
        // Récupérer les valeurs des paramètres
        const useWeightedPagerank = document.getElementById('use-weighted-pagerank').checked;
        const contentLinksOnly = document.getElementById('content-links-only').checked;
        const alpha = parseFloat(document.getElementById('alpha-coefficient').value);
        const beta = parseFloat(document.getElementById('beta-coefficient').value);
        
        // Mettre à jour les variables globales
        pagerankConfig.useWeightedPagerank = useWeightedPagerank;
        pagerankConfig.contentLinksOnly = contentLinksOnly;
        pagerankConfig.alpha = alpha;
        pagerankConfig.beta = beta;
        
        // Afficher un message de chargement
        showNotification('Calcul du PageRank en cours...', 'info');
        
        console.log(`Calcul du PageRank avec: useWeightedPagerank=${useWeightedPagerank}, contentLinksOnly=${contentLinksOnly}, alpha=${alpha}, beta=${beta}`);
        
        // Vider les données existantes pour forcer un rechargement complet
        pagerankData = null;
        pagerankTableData = [];
        
        // Recharger les données de PageRank avec la nouvelle configuration
        await loadPagerankData(jobId);
        
        // Réinitialiser les visualisations pour prendre en compte les nouvelles données
        if (currentNetwork) {
            currentNetwork.destroy();
            currentNetwork = null;
        }
        if (optimizedNetwork) {
            optimizedNetwork.destroy();
            optimizedNetwork = null;
        }
        
        // Recréer les visualisations avec les nouvelles données
        createCurrentVisualization();
        createOptimizedVisualization();
        
        // Mettre à jour les statistiques et le tableau PageRank
        displayStatistics();
        displayPagerankTable();
        
        // Afficher un message de succès
        const message = useWeightedPagerank 
            ? `PageRank pondéré appliqué (α=${alpha}, β=${beta})${contentLinksOnly ? ', liens de contenu uniquement' : ''}` 
            : `PageRank standard${contentLinksOnly ? ' (liens de contenu uniquement)' : ''}`;
        
        showNotification(message, 'success');
    } catch (error) {
        console.error('Erreur lors de l\'application du PageRank:', error);
        showNotification('Erreur lors du calcul du PageRank', 'danger');
    }
}

/**
 * Récupère l'ID de la tâche depuis l'URL
 * @returns {string|null} - ID de la tâche ou null si non trouvé
 */
function getJobIdFromUrl() {
    const pathParts = window.location.pathname.split('/');
    // L'ID de la tâche est généralement le dernier segment de l'URL
    const jobId = pathParts[pathParts.length - 1];
    return jobId || null;
}

/**
 * Affiche une notification temporaire
 * @param {string} message - Message à afficher
 * @param {string} type - Type de notification (info, success, warning, danger)
 */
function showNotification(message, type = 'info') {
    const notificationContainer = document.getElementById('notification-container');
    if (!notificationContainer) {
        // Créer le conteneur s'il n'existe pas
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    // Créer la notification
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close" onclick="this.parentElement.parentElement.parentElement.remove()"></button>
    `;
    
    // Ajouter la notification au conteneur
    document.getElementById('notification-container').appendChild(notification);
    
    // Supprimer automatiquement après 3 secondes
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}
