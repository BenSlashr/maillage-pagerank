/**
 * Visualisation de graphe avec Cytoscape.js
 * Une alternative plus performante pour les grands graphes
 */

// Configuration globale
const cyConfig = {
    // Couleurs pour les différents types de nœuds et d'arêtes
    colors: {
        nodes: {
            default: '#6c757d',
            category: '#007bff',
            product: '#28a745',
            blog: '#fd7e14',
            homepage: '#dc3545'
        },
        edges: {
            existing: '#6c757d',
            suggested: '#28a745'
        }
    },
    // Taille des nœuds
    nodeSize: {
        min: 5,
        max: 30,
        factor: 1000
    },
    // Mise en page
    layout: {
        name: 'cose',
        idealEdgeLength: 100,
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 30,
        randomize: false,
        componentSpacing: 100,
        nodeRepulsion: 400000,
        edgeElasticity: 100,
        nestingFactor: 5,
        gravity: 80,
        numIter: 1000,
        initialTemp: 200,
        coolingFactor: 0.95,
        minTemp: 1.0
    }
};

// Variables globales
let cyCurrentNetwork = null;
let cyOptimizedNetwork = null;
let graphData = null;
let pagerankData = null;
let visualizationConfig = null;

// Configuration pour le filtrage des nœuds
let nodeDisplayConfig = {
    maxNodes: 100,  // Nombre maximum de nœuds à afficher par défaut
    filterBy: 'pagerank'  // Critère de filtrage ('pagerank', 'links', 'all')
};

/**
 * Initialise la visualisation Cytoscape
 * @param {Object} data - Données du graphe
 */
function initCytoscapeVisualization(data) {
    console.log('Initialisation de la visualisation Cytoscape avec les données:', data);
    
    // Vérifier que les données sont valides
    if (!data || !data.graph) {
        console.error('Données de graphe manquantes');
        showError('Les données de graphe sont manquantes ou invalides.');
        return;
    }
    
    graphData = data.graph;
    pagerankData = data.pagerank;
    visualizationConfig = data.config || {
        use_weighted_pagerank: false,
        content_links_only: false,
        alpha: 0.5,
        beta: 0.5
    };
    
    // Vérifier que les donnœuds sont disponibles
    if (!graphData.nodes || graphData.nodes.length === 0) {
        console.error('Aucun nœud dans les données du graphe');
        showError('Aucune page HTML n\'a été trouvée dans les données. Vérifiez que vos URLs correspondent à des pages HTML.');
        return;
    }
    
    // Vérifier que les arêtes sont disponibles
    if (!graphData.edges) {
        console.error('Structure d\'arêtes manquante');
        graphData.edges = { current: [], suggested: [] };
    }
    
    if (!graphData.edges.current) {
        console.warn('Aucune arête existante dans les données');
        graphData.edges.current = [];
    }
    
    if (!graphData.edges.suggested) {
        console.warn('Aucune arête suggérée dans les données');
        graphData.edges.suggested = [];
    }
    
    // Initialiser les réseaux
    initCytoscapeNetwork('cy-current-graph', 'current');
    initCytoscapeNetwork('cy-optimized-graph', 'optimized');
    
    // Afficher les statistiques
    displayNetworkStats();
}

/**
 * Affiche un message d'erreur dans l'interface
 * @param {string} message - Message d'erreur
 */
function showError(message) {
    // Afficher l'erreur dans les conteneurs de graphe
    const containers = ['cy-current-graph', 'cy-optimized-graph'];
    
    containers.forEach(containerId => {
        const container = document.getElementById(containerId);
        if (container) {
            // Masquer le spinner
            const spinner = document.getElementById(`${containerId.replace('graph', 'spinner')}`);
            if (spinner) {
                spinner.style.display = 'none';
            }
            
            // Afficher le message d'erreur
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger m-3';
            errorDiv.innerHTML = `<strong>Erreur:</strong> ${message}`;
            container.appendChild(errorDiv);
        }
    });
    
    // Afficher l'erreur dans les statistiques
    const statsContainer = document.getElementById('network-stats');
    if (statsContainer) {
        statsContainer.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h5 class="mb-0">Statistiques du réseau</h5>
                </div>
                <div class="card-body">
                    <div class="alert alert-danger">
                        <strong>Erreur:</strong> ${message}
                    </div>
                </div>
            </div>
        `;
    }
}

/**
 * Initialise un réseau Cytoscape
 * @param {string} containerId - ID du conteneur
 * @param {string} mode - Mode de visualisation ('current' ou 'optimized')
 */
function initCytoscapeNetwork(containerId, mode) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Préparer les éléments
    const elements = prepareCytoscapeElements(mode);
    
    // Créer l'instance Cytoscape
    const cy = cytoscape({
        container: container,
        elements: elements,
        style: [
            // Style des nœuds
            {
                selector: 'node',
                style: {
                    'background-color': 'data(color)',
                    'label': 'data(label)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'font-size': '10px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-outline-width': 2,
                    'text-outline-color': '#fff',
                    'color': '#000'
                }
            },
            // Style des arêtes
            {
                selector: 'edge',
                style: {
                    'width': 1,
                    'line-color': 'data(color)',
                    'target-arrow-color': 'data(color)',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            },
            // Style des arêtes suggérées
            {
                selector: 'edge[suggested]',
                style: {
                    'line-style': 'dashed',
                    'line-dash-pattern': [6, 3]
                }
            }
        ],
        layout: cyConfig.layout
    });
    
    // Ajouter des événements
    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        showNodeDetails(node.data());
    });
    
    // Stocker l'instance
    if (mode === 'current') {
        cyCurrentNetwork = cy;
    } else {
        cyOptimizedNetwork = cy;
    }
    
    // Ajouter des contrôles de zoom
    addZoomControls(containerId, cy);
    
    return cy;
}

/**
 * Prépare les éléments pour Cytoscape
 * @param {string} mode - Mode de visualisation ('current' ou 'optimized')
 * @returns {Array} - Éléments pour Cytoscape
 */
function prepareCytoscapeElements(mode) {
    if (!graphData || !graphData.nodes) return [];
    
    // Filtrer les nœuds par PageRank
    let nodes = [...graphData.nodes];
    if (nodeDisplayConfig.maxNodes > 0 && nodeDisplayConfig.maxNodes < nodes.length) {
        // Ajouter le PageRank à chaque nœud
        nodes = nodes.map(node => {
            let pagerank = 0;
            if (pagerankData && pagerankData.current && node.id in pagerankData.current) {
                pagerank = pagerankData.current[node.id].pagerank;
            }
            return { ...node, pagerank };
        });
        
        // Trier et filtrer
        nodes = nodes.sort((a, b) => b.pagerank - a.pagerank).slice(0, nodeDisplayConfig.maxNodes);
    }
    
    // Créer un Set des IDs de nœuds pour un filtrage efficace
    const nodeIds = new Set(nodes.map(node => node.id));
    
    // Préparer les nœuds pour Cytoscape
    const cyNodes = nodes.map(node => {
        // Déterminer la taille du nœud en fonction du PageRank
        let size = cyConfig.nodeSize.min;
        if (pagerankData && pagerankData.current && node.id in pagerankData.current) {
            const pagerank = pagerankData.current[node.id].pagerank;
            size = Math.max(
                cyConfig.nodeSize.min,
                Math.min(cyConfig.nodeSize.max, pagerank * cyConfig.nodeSize.factor)
            );
        }
        
        // Déterminer la couleur en fonction du type
        let color = cyConfig.colors.nodes.default;
        if (node.type in cyConfig.colors.nodes) {
            color = cyConfig.colors.nodes[node.type];
        }
        
        return {
            data: {
                id: node.id,
                label: node.label || node.id.split('/').pop(),
                color: color,
                size: size,
                type: node.type,
                originalData: node
            }
        };
    });
    
    // Préparer les arêtes pour Cytoscape
    let cyEdges = [];
    
    // Ajouter les liens existants (uniquement entre nœuds visibles)
    if (graphData.edges.current) {
        const currentEdges = graphData.edges.current
            .filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target))
            .map(edge => ({
                data: {
                    id: `e-${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    color: cyConfig.colors.edges.existing,
                    originalData: edge
                }
            }));
        
        cyEdges = cyEdges.concat(currentEdges);
    }
    
    // Ajouter les liens suggérés si on est en mode optimisé
    if (mode === 'optimized' && graphData.edges.suggested) {
        const suggestedEdges = graphData.edges.suggested
            .filter(edge => nodeIds.has(edge.source) && nodeIds.has(edge.target))
            .map(edge => ({
                data: {
                    id: `s-${edge.source}-${edge.target}`,
                    source: edge.source,
                    target: edge.target,
                    color: cyConfig.colors.edges.suggested,
                    suggested: true,
                    anchor: edge.anchor || '',
                    originalData: edge
                }
            }));
        
        cyEdges = cyEdges.concat(suggestedEdges);
    }
    
    console.log(`Cytoscape: ${cyNodes.length} nœuds et ${cyEdges.length} liens préparés pour le mode ${mode}`);
    return [...cyNodes, ...cyEdges];
}

/**
 * Ajoute des contrôles de zoom à un graphe Cytoscape
 * @param {string} containerId - ID du conteneur
 * @param {Object} cy - Instance Cytoscape
 */
function addZoomControls(containerId, cy) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Créer les boutons
    const controlsDiv = document.createElement('div');
    controlsDiv.className = 'cy-controls';
    controlsDiv.innerHTML = `
        <button class="cy-zoom-in" title="Zoom in">+</button>
        <button class="cy-zoom-out" title="Zoom out">-</button>
        <button class="cy-zoom-reset" title="Reset zoom">⟲</button>
    `;
    container.appendChild(controlsDiv);
    
    // Ajouter les événements
    container.querySelector('.cy-zoom-in').addEventListener('click', () => {
        cy.zoom(cy.zoom() * 1.2);
        cy.center();
    });
    
    container.querySelector('.cy-zoom-out').addEventListener('click', () => {
        cy.zoom(cy.zoom() / 1.2);
        cy.center();
    });
    
    container.querySelector('.cy-zoom-reset').addEventListener('click', () => {
        cy.fit();
    });
}

/**
 * Affiche les détails d'un nœud
 * @param {Object} nodeData - Données du nœud
 */
function showNodeDetails(nodeData) {
    const detailsContainer = document.getElementById('node-details');
    if (!detailsContainer) return;
    
    // Préparer les détails de PageRank
    let pagerankDetails = '';
    if (pagerankData && pagerankData.current && nodeData.id in pagerankData.current) {
        const current = pagerankData.current[nodeData.id];
        pagerankDetails += `
            <div class="mb-3">
                <h6>PageRank actuel</h6>
                <p>Score: ${current.pagerank.toFixed(6)}</p>
                <p>Rang: ${current.rank} sur ${Object.keys(pagerankData.current).length}</p>
            </div>
        `;
        
        if (pagerankData.optimized && nodeData.id in pagerankData.optimized) {
            const optimized = pagerankData.optimized[nodeData.id];
            const improvement = pagerankData.improvement[nodeData.id];
            
            const improvementClass = improvement.percentage > 0 ? 'text-success' : 
                                   improvement.percentage < 0 ? 'text-danger' : '';
            
            const rankChangeClass = improvement.rank_change > 0 ? 'text-success' : 
                                  improvement.rank_change < 0 ? 'text-danger' : '';
            
            const rankChangeText = improvement.rank_change > 0 ? `+${improvement.rank_change}` : 
                                 improvement.rank_change < 0 ? `${improvement.rank_change}` : '0';
            
            pagerankDetails += `
                <div class="mb-3">
                    <h6>PageRank optimisé</h6>
                    <p>Score: ${optimized.pagerank.toFixed(6)}</p>
                    <p>Rang: ${optimized.rank} sur ${Object.keys(pagerankData.optimized).length}</p>
                    <p>Amélioration: <span class="${improvementClass}">${improvement.percentage.toFixed(2)}%</span></p>
                    <p>Changement de rang: <span class="${rankChangeClass}">${rankChangeText}</span></p>
                </div>
            `;
        }
    }
    
    // Afficher les détails
    detailsContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Détails de la page</h5>
            </div>
            <div class="card-body">
                <h6 class="mb-3">${nodeData.id}</h6>
                
                <div class="mb-3">
                    <strong>Type:</strong> ${nodeData.type || 'Non spécifié'}
                </div>
                
                ${pagerankDetails}
                
                <div class="mt-3">
                    <button class="btn btn-sm btn-primary" onclick="focusOnNode('${nodeData.id}')">
                        Centrer sur cette page
                    </button>
                </div>
            </div>
        </div>
    `;
}

/**
 * Centre la vue sur un nœud spécifique
 * @param {string} nodeId - ID du nœud
 */
function focusOnNode(nodeId) {
    const activeTab = document.querySelector('.tab-pane.active');
    if (!activeTab) return;
    
    let cy = null;
    if (activeTab.id === 'cy-current-network') {
        cy = cyCurrentNetwork;
    } else if (activeTab.id === 'cy-optimized-network') {
        cy = cyOptimizedNetwork;
    }
    
    if (cy) {
        const node = cy.getElementById(nodeId);
        if (node.length > 0) {
            cy.fit(node, 50);
            cy.center(node);
        }
    }
}

/**
 * Met à jour le nombre maximum de nœuds à afficher
 * @param {number} newMaxNodes - Nouveau nombre maximum de nœuds
 */
function updateCyMaxNodes(newMaxNodes) {
    nodeDisplayConfig.maxNodes = newMaxNodes;
    
    // Mettre à jour les visualisations
    if (cyCurrentNetwork) {
        const elements = prepareCytoscapeElements('current');
        cyCurrentNetwork.elements().remove();
        cyCurrentNetwork.add(elements);
        cyCurrentNetwork.layout(cyConfig.layout).run();
    }
    
    if (cyOptimizedNetwork) {
        const elements = prepareCytoscapeElements('optimized');
        cyOptimizedNetwork.elements().remove();
        cyOptimizedNetwork.add(elements);
        cyOptimizedNetwork.layout(cyConfig.layout).run();
    }
    
    // Mettre à jour les statistiques
    displayNetworkStats();
}

/**
 * Affiche les statistiques du réseau
 */
function displayNetworkStats() {
    const statsContainer = document.getElementById('network-stats');
    if (!statsContainer || !graphData) return;
    
    const totalNodes = graphData.nodes.length;
    const displayedNodes = nodeDisplayConfig.maxNodes > 0 ? 
                          Math.min(nodeDisplayConfig.maxNodes, totalNodes) : 
                          totalNodes;
    
    const totalCurrentEdges = graphData.edges.current ? graphData.edges.current.length : 0;
    const totalSuggestedEdges = graphData.edges.suggested ? graphData.edges.suggested.length : 0;
    
    // Information sur le type de PageRank utilisé
    const pagerankType = visualizationConfig?.use_weighted_pagerank ? 'pondéré' : 'standard';
    const contentLinksOnly = visualizationConfig?.content_links_only ? 'Oui' : 'Non';
    
    // Informations supplémentaires pour le PageRank pondéré
    let pagerankInfo = '';
    if (visualizationConfig?.use_weighted_pagerank) {
        pagerankInfo = `
            <div class="alert alert-info mt-3">
                <h6 class="alert-heading">Configuration du PageRank pondéré</h6>
                <p class="mb-1"><strong>Coefficient sémantique (α):</strong> ${visualizationConfig.alpha.toFixed(1)}</p>
                <p class="mb-1"><strong>Coefficient position (β):</strong> ${visualizationConfig.beta.toFixed(1)}</p>
                <p class="mb-0"><small>Un coefficient plus élevé donne plus d'importance au facteur correspondant</small></p>
            </div>
        `;
    }
    
    statsContainer.innerHTML = `
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Statistiques du réseau</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6>Structure du site</h6>
                        <p><strong>Pages totales:</strong> ${totalNodes}</p>
                        <p><strong>Pages affichées:</strong> ${displayedNodes} (${Math.round(displayedNodes/totalNodes*100)}%)</p>
                        <p><strong>Liens actuels:</strong> ${totalCurrentEdges}</p>
                        <p><strong>Liens suggérés:</strong> ${totalSuggestedEdges}</p>
                        <p><strong>Densité du réseau:</strong> ${(totalCurrentEdges / (totalNodes * (totalNodes - 1))).toFixed(5)}</p>
                    </div>
                    <div class="col-md-6">
                        <h6>Configuration PageRank</h6>
                        <p><strong>Type de PageRank:</strong> ${pagerankType}</p>
                        <p><strong>Liens de contenu uniquement:</strong> ${contentLinksOnly}</p>
                        ${pagerankInfo}
                    </div>
                </div>
            </div>
        </div>
    `;
}

/**
 * Change le layout du graphe
 * @param {string} layoutName - Nom du layout
 */
function changeLayout(layoutName) {
    let layoutConfig = { ...cyConfig.layout, name: layoutName };
    
    // Ajuster les paramètres en fonction du layout
    switch (layoutName) {
        case 'cose':
            // Configuration par défaut
            break;
        case 'concentric':
            layoutConfig = {
                name: 'concentric',
                concentric: function(node) {
                    return node.data('size');
                },
                levelWidth: function() {
                    return 2;
                },
                minNodeSpacing: 50
            };
            break;
        case 'breadthfirst':
            layoutConfig = {
                name: 'breadthfirst',
                directed: true,
                spacingFactor: 1.5
            };
            break;
        case 'grid':
            layoutConfig = {
                name: 'grid',
                avoidOverlap: true,
                spacingFactor: 1.5
            };
            break;
        case 'circle':
            layoutConfig = {
                name: 'circle',
                avoidOverlap: true,
                spacingFactor: 1.5
            };
            break;
    }
    
    // Appliquer le nouveau layout
    const activeTab = document.querySelector('.tab-pane.active');
    if (!activeTab) return;
    
    let cy = null;
    if (activeTab.id === 'cy-current-network') {
        cy = cyCurrentNetwork;
    } else if (activeTab.id === 'cy-optimized-network') {
        cy = cyOptimizedNetwork;
    }
    
    if (cy) {
        cy.layout(layoutConfig).run();
    }
}

// Exporter les fonctions
window.initCytoscapeVisualization = initCytoscapeVisualization;
window.updateCyMaxNodes = updateCyMaxNodes;
window.changeLayout = changeLayout;
